import gzip
import os
import pickle

import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.dataAugmentation import dataAugmentation
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration


class compileModelDataHelpers:
    def __init__(self, submodel, userInputParams, accelerator=None):
        # General parameters
        self.compiledInfoLocation = os.path.dirname(__file__) + "/../../../_experimentalData/_compiledData/"
        self.userInputParams = userInputParams
        self.missingLabelValue = torch.nan
        self.compiledExtension = ".pkl.gz"
        self.accelerator = accelerator

        # Make Output Folder Directory if Not Already Created
        os.makedirs(self.compiledInfoLocation, exist_ok=True)

        # Initialize relevant classes.
        self.modelParameters = modelParameters(accelerator)
        self.modelMigration = modelMigration(accelerator, debugFlag=False)
        self.dataAugmentation = dataAugmentation()
        self.generalMethods = generalMethods()

        # Submodel-specific parameters
        self.emotionPredictionModelInfo = None
        self.signalEncoderModelInfo = None
        self.minSignalPresentCount = None
        self.maxSinglePointDiff = None
        self.minSequencePoints = None
        self.minBoundaryPoints = None
        self.maxAverageDiff = None

        # Exclusion criterion.
        self.minSequencePoints, self.minSignalPresentCount, self.minBoundaryPoints, self.maxSinglePointDiff, self.maxAverageDiff = self.modelParameters.getExclusionSequenceCriteria()
        self.adaptiveFactor = self.modelParameters.getAdaptiveFactor()

    # ---------------------- Model Specific Parameters --------------------- #

    @staticmethod
    def embedInformation(submodel, userInputParams, trainingDate):
        # Embedded information for each model.
        signalEncoderModelInfo = f"signalEncoder on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} at sigEncLayers {userInputParams['numSignalEncoderLayers']} at goldenRatio {userInputParams['goldenRatio']} at encodedDim {userInputParams['encodedDimension']} at lr {userInputParams['learningRate']} at wd {userInputParams['weightDecay']} with wave {userInputParams['waveletType']}"
        emotionPredictionModelInfo = f"emotionPrediction on {userInputParams['deviceListed']} with {userInputParams['optimizerType']}"

        if submodel == modelConstants.signalEncoderModel:
            return f"{trainingDate} {signalEncoderModelInfo.replace(".", "-")}"
        elif submodel == modelConstants.emotionModel:
            return f"{trainingDate} {emotionPredictionModelInfo.replace(".", "-")}"
        else: raise Exception()

    # ---------------------- Saving/Loading Model Data --------------------- #

    def saveCompiledInfo(self, data_to_store, saveDataName):
        with gzip.open(filename=f'{self.compiledInfoLocation}{saveDataName}{self.compiledExtension}', mode='wb') as file:
            pickle.dump(data_to_store, file)

    def loadCompiledInfo(self, loadDataName):
        with gzip.open(filename=f'{self.compiledInfoLocation}{loadDataName}{self.compiledExtension}', mode='rb') as file:
            data_loaded = pickle.load(file)
        return data_loaded[f"{loadDataName}"]

    # ------------------------- Signal Organization ------------------------ #

    @staticmethod
    def organizeActivityLabels(activityNames, allFeatureLabels, activityLabelInd):
        # Remove the bad or unknown labels.
        goodActivityMask = ~torch.isnan(allFeatureLabels[:, activityLabelInd])
        activityLabels = allFeatureLabels[:, activityLabelInd][goodActivityMask]

        # Find the unique activity labels
        uniqueActivityLabels, validActivityLabels = torch.unique(activityLabels, return_inverse=True)
        assert len(activityLabels) == len(validActivityLabels), f"{len(activityLabels)} != {len(validActivityLabels)}"

        # Get the corresponding unique activity names
        uniqueActivityNames = np.asarray(activityNames)[uniqueActivityLabels.to(torch.int)]
        allFeatureLabels[:, activityLabelInd][goodActivityMask] = validActivityLabels.to(allFeatureLabels.dtype)

        return uniqueActivityNames, allFeatureLabels

    def _preprocessLabels(self, allFeatureLabels):
        # allFeatureLabels: A torch array or list of size (batchSize, numLabels)
        # metaTraining: Boolean indicating if the data is for training
        # Convert to tensor and initialize lists
        batchSize, numLabels = allFeatureLabels.shape
        allSingleClassMasks = []

        # Mask out the bad or unknown labels.
        goodLabelInds = 0 <= allFeatureLabels  # The minimum label should be 0
        allFeatureLabels[~goodLabelInds] = self.missingLabelValue

        # For each label type.
        for labelTypeInd in range(numLabels):
            featureLabels = allFeatureLabels[:, labelTypeInd].clone()
            goodLabels = goodLabelInds[:, labelTypeInd]
            featureLabels = featureLabels.int()

            # Count the number of times the emotion label has a unique value.
            unique_classes, class_counts = torch.unique(featureLabels[goodLabels], return_counts=True)
            smallClassMask = class_counts <= 2

            # Remove labels belonging to small classes.
            smallClassLabelMask = torch.isin(featureLabels, unique_classes[smallClassMask])
            allSingleClassMasks.append(smallClassLabelMask)

        return allFeatureLabels, allSingleClassMasks

    @staticmethod
    def addContextualInfo(allSignalData, allNumSignalPoints, allSubjectInds, datasetInd):
        # allSignalData: A torch tensor of size (batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel])
        # allNumSignalPoints: A torch array of size (batchSize, numSignals)
        # allSubjectInds: A torch array of size batchSize
        numExperiments, numSignals, maxSequenceLength, numChannels = allSignalData.shape
        numSignalIdentifiers = len(modelConstants.signalIdentifiers)
        numMetadata = len(modelConstants.metadata)

        # Create lists to store the new augmented data.
        compiledSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength + numSignalIdentifiers + numMetadata, numChannels))
        assert len(modelConstants.signalChannelNames) == numChannels

        # For each recorded experiment.
        for experimentInd in range(numExperiments):
            # Compile all the metadata information: dataset specific.
            subjectInds = torch.full(size=(numSignals, 1, numChannels), fill_value=allSubjectInds[experimentInd])
            datasetInds = torch.full(size=(numSignals, 1, numChannels), fill_value=datasetInd)
            metadata = torch.hstack((datasetInds, subjectInds))
            # metadata dim: numSignals, numMetadata, numChannels

            # Compile all the signal information: signal specific.
            eachSignal_numPoints = allNumSignalPoints[experimentInd].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, numChannels)
            signalInds = torch.arange(numSignals).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, numChannels)
            batchInds = torch.full(size=(numSignals, 1, numChannels), fill_value=experimentInd)
            signalIdentifiers = torch.hstack((eachSignal_numPoints, signalInds, batchInds))
            # signalIdentifiers dim: numSignals, numSignalIdentifiers, numChannels

            # Assert the correct hardcoded dimensions.
            assert emotionDataInterface.getSignalIdentifierIndex(identifierName=modelConstants.numSignalPointsSI) == 0, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getSignalIdentifierIndex(identifierName=modelConstants.signalIndexSI) == 1, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getMetadataIndex(metadataName=modelConstants.datasetIndexMD) == 0, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getMetadataIndex(metadataName=modelConstants.subjectIndexMD) == 1, "Asserting I am self-consistent. Hardcoded assertion"
            assert numSignalIdentifiers == 3, "Asserting I am self-consistent. Hardcoded assertion"
            assert numMetadata == 2, "Asserting I am self-consistent. Hardcoded assertion"

            # Add the demographic data to the feature array.
            compiledSignalData[experimentInd] = torch.hstack((allSignalData[experimentInd], signalIdentifiers, metadata))

        return compiledSignalData

    # ---------------------------------------------------------------------- #
    # ---------------------------- Data Cleaning --------------------------- #

    @staticmethod
    def _padSignalData(allRawFeatureIntervalTimes, allRawFeatureIntervals, referenceTimes):
        # allRawFeatureIntervals: batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
        # allRawFeatureIntervalTimes: batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant
        # referenceTimes: A list of size (batchSize)
        # Determine the final dimensions of the padded array.
        maxSequenceLength = max(max(len(biomarkerTimes) for biomarkerTimes in experimentalTimes) for experimentalTimes in allRawFeatureIntervalTimes)
        numSignals = sum(len(biomarkerData[0]) for biomarkerData in allRawFeatureIntervals[0])
        numExperiments = len(allRawFeatureIntervals)

        # Initialize the padded array and end signal indices list
        allSignalData = torch.zeros(size=(numExperiments, numSignals, maxSequenceLength, len(modelConstants.signalChannelNames)))  # +1 for the time data
        allNumSignalPoints = torch.empty(size=(numExperiments, numSignals), dtype=torch.int)

        # Get the indices for each of the signal information.
        dataChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.signalChannel)
        timeChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.timeChannel)

        maxSequenceLength = 0
        # For each batch of biomarkers.
        for experimentalInd in range(numExperiments):
            batchTimes = allRawFeatureIntervalTimes[experimentalInd]
            batchData = allRawFeatureIntervals[experimentalInd]
            surveyAnswerTime = referenceTimes[experimentalInd]

            currentSignalInd = 0
            # For each biomarker in the batch.
            for (biomarkerData, biomarkerTimes) in zip(batchData, batchTimes):
                biomarkerData = torch.tensor(biomarkerData).T  # Dim: numBiomarkerFeatures, batchSpecificFeatureLength
                biomarkerTimes = torch.tensor(biomarkerTimes)  # Dim: batchSpecificFeatureLength
                biomarkerTimes = surveyAnswerTime - biomarkerTimes

                # Remove data outside the time window.
                timeWindowMask = (biomarkerTimes <= modelConstants.timeWindows[-1]).to(torch.bool)
                biomarkerData = biomarkerData[:, timeWindowMask]
                biomarkerTimes = biomarkerTimes[timeWindowMask]

                # Get the number of signals in the current biomarker.
                numBiomarkerFeatures, batchSpecificFeatureLength = biomarkerData.shape
                finalSignalInd = currentSignalInd + numBiomarkerFeatures

                # Fill the padded array with the signal data

                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0:batchSpecificFeatureLength, timeChannelInd] = biomarkerTimes
                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0:batchSpecificFeatureLength, dataChannelInd] = biomarkerData
                allNumSignalPoints[experimentalInd, currentSignalInd:finalSignalInd] = batchSpecificFeatureLength

                # Update the current signal index
                maxSequenceLength = max(maxSequenceLength, batchSpecificFeatureLength)
                currentSignalInd = finalSignalInd

        # Remove unused points.
        allSignalData = allSignalData[:, :, 0:maxSequenceLength, :]

        return allSignalData, allNumSignalPoints

    def _preprocessSignals(self, allSignalData, allNumSignalPoints, featureNames):
        # allSignalData: A torch array of size (batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel])
        # allNumSignalPoints: A torch array of size (batchSize, numSignals)
        # featureNames: A torch array of size numSignals
        # Ensure the feature names match the number of signals
        assert len(allSignalData) == 0 or len(featureNames) == allSignalData.shape[1], \
            f"Feature names do not match data dimensions. {len(featureNames)} != {allSignalData.shape[1]}"

        # Standardize all signals at once for the entire batch
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData)
        allSignalData = self.normalizeSignals(allSignalData=allSignalData, missingDataMask=~validDataMask)
        biomarkerData = emotionDataInterface.getChannelData(signalData=allSignalData, channelName=modelConstants.signalChannel)
        # allSignalData dim: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # missingDataMask dim: batchSize, numSignals, maxSequenceLength
        # biomarkerData: batchSize, numSignals, maxSequenceLength

        # Find single point differences
        badSinglePointMaxDiff = self.maxSinglePointDiff < biomarkerData.diff(dim=-1).abs()  # Maximum difference between consecutive points: batchSize, numSignals, maxSequenceLength-1
        allSignalData[:, :, :-1][badSinglePointMaxDiff] = 0  # Remove small errors.
        allSignalData[:, :, 1:][badSinglePointMaxDiff] = 0  # Remove small errors.

        # Find single boundary points.
        boundaryPointsMask = modelConstants.minMaxScale - 0.25 < biomarkerData  # batchSize, numSignals, maxSequenceLength
        goodBatchSignalBoundaries = 2 <= boundaryPointsMask.sum(dim=-1)  # batchSize, numSignals
        boundaryPointsMask[goodBatchSignalBoundaries.unsqueeze(-1).expand_as(boundaryPointsMask)] = False
        allSignalData[boundaryPointsMask.unsqueeze(-1).expand_as(allSignalData)] = 0

        # Find single boundary points.
        boundaryPointsMask = biomarkerData < -modelConstants.minMaxScale + 0.25  # batchSize, numSignals, maxSequenceLength
        goodBatchSignalBoundaries = 2 <= boundaryPointsMask.sum(dim=-1)  # batchSize, numSignals
        boundaryPointsMask[goodBatchSignalBoundaries.unsqueeze(-1).expand_as(boundaryPointsMask)] = False
        allSignalData[boundaryPointsMask.unsqueeze(-1).expand_as(allSignalData)] = 0

        # Re-normalize the data after removing bad points.
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData)
        allSignalData = self.normalizeSignals(allSignalData=allSignalData, missingDataMask=~validDataMask)
        biomarkerData = emotionDataInterface.getChannelData(signalData=allSignalData, channelName=modelConstants.signalChannel)
        biomarkerDiffs = biomarkerData.diff(dim=-1).abs()
        biomarkerDiffs[biomarkerDiffs == 0] = torch.nan

        # Create boolean masks for signals that don’t meet the requirements
        minLowerBoundaryMask = self.minBoundaryPoints <= (biomarkerData[:, :, 1:-1] < -modelConstants.minMaxScale + 0.25).sum(dim=-1)  # Number of points below -0.95: batchSize, numSignals
        minUpperBoundaryMask = self.minBoundaryPoints <= (modelConstants.minMaxScale - 0.25 < biomarkerData[:, :, 1:-1]).sum(dim=-1)  # Number of points above 0.95: batchSize, numSignals
        averageDiff = biomarkerDiffs.nanmean(dim=-1) < self.maxAverageDiff  # Average difference between consecutive points: batchSize, numSignals
        minPointsMask = self.minSequencePoints <= validDataMask.sum(dim=-1)  # Minimum number of points: batchSize, numSignals
        validSignalMask = validDataMask.any(dim=-1)  # Missing data: batchSize, numSignals

        # Combine all masks into a single mask and expand to match dimensions.
        validSignalMask = minPointsMask & minLowerBoundaryMask & minUpperBoundaryMask & averageDiff & validSignalMask
        validSignalInds = self.minSignalPresentCount < validSignalMask.sum(dim=0)

        # Filter out the invalid signals
        allSignalData[~validSignalMask.unsqueeze(-1).unsqueeze(-1).expand_as(allSignalData)] = 0
        allNumSignalPoints[~validSignalMask] = 0

        return allSignalData[:, validSignalInds, :, :], allNumSignalPoints[:, validSignalInds], featureNames[validSignalInds]

    def normalizeSignals(self, allSignalData, missingDataMask):
        # signalBatchData dimension: numExperiments, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # missingDataMask dimension: numExperiments, numSignals, maxSequenceLength
        signalChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.signalChannel)
        allSignalData[:, :, :, signalChannelInd] = self.generalMethods.minMaxScale_noInverse(allSignalData[:, :, :, signalChannelInd], scale=modelConstants.minMaxScale, missingDataMask=missingDataMask)

        return allSignalData

    def preprocessingSignalsTherapy(self, allSignalData, allNumSignalPoints):
        # allSignalData: A torch array of size (batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel])
        # allNumSignalPoints: A torch array of size (batchSize, numSignals)
        # featureNames: A torch array of size numSignals
        # Ensure the feature names match the number of signals
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData)
        allSignalData = self.normalizeSignals(allSignalData=allSignalData, missingDataMask=~validDataMask)
        biomarkerData = emotionDataInterface.getChannelData(signalData=allSignalData, channelName=modelConstants.signalChannel)
        allSignalData = self.adaptivePreprocessSignal(allSignalData, biomarkerData)

        # Re-normalize the data after removing bad points.
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData)
        allSignalData = self.normalizeSignals(allSignalData=allSignalData, missingDataMask=~validDataMask)
        biomarkerData = emotionDataInterface.getChannelData(signalData=allSignalData, channelName=modelConstants.signalChannel)

        # Create boolean masks for signals that don’t meet the requirements
        minLowerBoundaryMask = self.minBoundaryPoints <= (biomarkerData < -modelConstants.minMaxScale + 0.25).sum(dim=-1)  # Number of points below -0.95: batchSize, numSignals
        minUpperBoundaryMask = self.minBoundaryPoints <= (modelConstants.minMaxScale - 0.25 < biomarkerData).sum(dim=-1)  # Number of points above 0.95: batchSize, numSignals
        averageDiff = biomarkerData.diff(dim=-1).abs().mean(dim=-1) <= self.maxAverageDiff  # Average difference between consecutive points: batchSize, numSignals
        minPointsMask = self.minSequencePoints <= validDataMask.sum(dim=-1)  # Minimum number of points: batchSize, numSignals
        validSignalMask = validDataMask.any(dim=-1)  # Missing data: batchSize, numSignals

        # Combine all masks into a single mask and expand to match dimensions.
        validSignalMask = minPointsMask & minLowerBoundaryMask & minUpperBoundaryMask & averageDiff & validSignalMask
        validSignalInds = self.minSignalPresentCount < validSignalMask.sum(dim=0)

        # Filter out the invalid signals
        allSignalData[~validSignalMask.unsqueeze(-1).unsqueeze(-1).expand_as(allSignalData)] = 0
        allNumSignalPoints[~validSignalMask] = 0

        return allSignalData[:, validSignalInds, :, :], allNumSignalPoints[:, validSignalInds]

    def adaptivePreprocessSignal(self, allSignalData, signalData):
        # adaptive filtering based on the general variability of the signal
        # if the signal has high variability, then the threshold is set higher to prevent loss of valid points
        # vice versa
        channel_std = signalData.std(dim=-1, keepdim=True)

        # TODO: finish implement this
        threshold = self.adaptiveFactor * channel_std
        diffMagnitude = signalData.diff(dim=-1).abs()
        badMaxSinglePointDiff = self.maxSinglePointDiff < diffMagnitude

        # Remove any bad data points and neighborring points
        allSignalData[:, :, :-1][badMaxSinglePointDiff] = 0
        allSignalData[:, :, 1:][badMaxSinglePointDiff] = 0

        return allSignalData
