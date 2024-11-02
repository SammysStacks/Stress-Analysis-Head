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
        self.maxClassPercentage = None
        self.minSequencePoints = None
        self.maxAverageDiff = None
        self.minNumClasses = None
        self.maxSinglePointDiff = None

        # Exclusion criterion.
        self.minNumClasses, self.maxClassPercentage = self.modelParameters.getExclusionClassCriteria(submodel)
        self.minSequencePoints, self.minSignalPresentCount, self.maxSinglePointDiff, self.maxAverageDiff = self.modelParameters.getExclusionSequenceCriteria(submodel)

    # ---------------------- Model Specific Parameters --------------------- #

    @staticmethod
    def embedInformation(submodel, userInputParams, trainingDate):
        # Embedded information for each model.
        signalEncoderModelInfo = f"signalEncoder on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} at numSignalEncoderLayers {userInputParams['numSignalEncoderLayers']} at goldenRatio {userInputParams['goldenRatio']} at encodedDimension {userInputParams['encodedDimension']}"
        emotionPredictionModelInfo = f"emotionPrediction on {userInputParams['deviceListed']} with {userInputParams['optimizerType']}"

        if submodel == modelConstants.signalEncoderModel:
            return f"{trainingDate} {signalEncoderModelInfo}"
        elif submodel == modelConstants.emotionModel:
            return f"{trainingDate} {emotionPredictionModelInfo}"
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
    def organizeActivityLabels(activityNames, activityLabels):
        # Find the unique activity labels
        uniqueActivityLabels, validActivityLabels = torch.unique(activityLabels, return_inverse=True)
        assert len(activityLabels) == len(validActivityLabels), f"{len(activityLabels)} != {len(validActivityLabels)}"

        # Get the corresponding unique activity names
        uniqueActivityNames = np.asarray(activityNames)[uniqueActivityLabels.to(torch.int)]

        return uniqueActivityNames, validActivityLabels.to(torch.float32)

    def organizeLabels(self, allFeatureLabels, metaTraining):
        # allFeatureLabels: A torch array or list of size (batchSize, numLabels)
        # metaTraining: Boolean indicating if the data is for training
        # Convert to tensor and initialize lists
        batchSize, numLabels = allFeatureLabels.shape
        allSingleClassIndices = [[] for _ in range(numLabels)]

        # Iterate over each label type (emotion)
        for labelTypeInd in range(numLabels):
            featureLabels = allFeatureLabels[:, labelTypeInd]
            # featureLabels dim: batchSize

            # Mask out unknown labels
            goodLabelInds = 0 <= featureLabels  # The minimum label should be 0
            featureLabels[~goodLabelInds] = self.missingLabelValue

            # Count the number of times the emotion label has a unique value.
            unique_classes, class_counts = torch.unique(featureLabels[goodLabelInds], return_counts=True)
            smallClassMask = class_counts < 3

            # For small distributions.
            if smallClassMask.any():
                # Remove labels belonging to small classes
                smallClassLabels = unique_classes[smallClassMask]
                smallClassLabelMask = torch.isin(featureLabels, smallClassLabels)
                allSingleClassIndices[labelTypeInd] = smallClassLabelMask.cpu().numpy().tolist()
                featureLabels[smallClassLabelMask] = self.missingLabelValue

                # Recalculate unique classes.
                goodLabelInds = goodLabelInds & ~smallClassLabelMask
                unique_classes, class_counts = torch.unique(featureLabels[goodLabelInds], return_counts=True)

            # Ensure greater variability in the class rating system.
            if metaTraining and (len(unique_classes) < self.minNumClasses or self.maxClassPercentage <= class_counts.max().item()/batchSize):
                featureLabels[:] = self.missingLabelValue

            # Save the edits made to the featureLabels
            allFeatureLabels[:, labelTypeInd] = featureLabels

        return allFeatureLabels, allSingleClassIndices

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
    def _padSignalData(allRawFeatureIntervalTimes, allRawFeatureIntervals, surveyAnswerTimes):
        # allRawFeatureIntervals : batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
        # allRawFeatureIntervalTimes : batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant
        # allSignalData : A list of size (batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel])
        # allNumSignalPoints : A list of size (batchSize, numSignals)
        # surveyAnswerTimes : A list of size (batchSize)
        # Determine the final dimensions of the padded array.
        maxSequenceLength = max(max(len(biomarkerTimes) for biomarkerTimes in experimentalTimes) for experimentalTimes in allRawFeatureIntervalTimes)
        numSignals = sum(len(biomarkerData[0]) for biomarkerData in allRawFeatureIntervals[0])
        numExperiments = len(allRawFeatureIntervals)

        # Initialize the padded array and end signal indices list
        allSignalData = torch.zeros(size=(numExperiments, numSignals, maxSequenceLength, len(modelConstants.signalChannelNames)), dtype=torch.float32)  # +1 for the time data
        allNumSignalPoints = torch.empty(size=(numExperiments, numSignals), dtype=torch.int)

        # Get the indices for each of the signal information.
        dataChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.signalChannel)
        timeChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.timeChannel)
        maxSequenceLength = 0

        # For each batch of biomarkers.
        for experimentalInd in range(numExperiments):
            batchData = allRawFeatureIntervals[experimentalInd]
            batchTimes = allRawFeatureIntervalTimes[experimentalInd]
            surveyAnswerTime = surveyAnswerTimes[experimentalInd]

            currentSignalInd = 0
            # For each biomarker in the batch.
            for (biomarkerData, biomarkerTimes) in zip(batchData, batchTimes):
                biomarkerData = torch.tensor(biomarkerData, dtype=torch.float32).T  # Dim: numBiomarkerFeatures, batchSpecificFeatureLength
                biomarkerTimes = torch.tensor(biomarkerTimes, dtype=torch.float32)  # Dim: batchSpecificFeatureLength
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
        batchSize, numSignals, maxSequenceLength, numChannels = allSignalData.shape

        # Standardize all signals at once for the entire batch
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData, allNumSignalPoints)
        allSignalData = self.normalizeSignals(allSignalData=allSignalData, missingDataMask=~validDataMask)
        biomarkerData = emotionDataInterface.getChannelData(signalData=allSignalData, channelName=modelConstants.signalChannel)
        # allSignalData dim: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # missingDataMask dim: batchSize, numSignals, maxSequenceLength
        # biomarkerData: batchSize, numSignals, maxSequenceLength

        biomarkerDiff = biomarkerData.diff(dim=-1).abs()
        # Create boolean masks for signals that don’t meet the requirements
        singlePointMaxDiff = ((biomarkerDiff[:, :, :-1] < self.maxSinglePointDiff) & (biomarkerDiff[:, :, 1:] < self.maxSinglePointDiff)).all(dim=-1)  # Maximum difference between consecutive points: batchSize, numSignals
        minLowerBoundaryMask = 2 < (biomarkerData < -modelConstants.minMaxScale + 0.3).sum(dim=-1)  # Number of points below -0.95: batchSize, numSignals
        minUpperBoundaryMask = 2 < (modelConstants.minMaxScale - 0.3 < biomarkerData).sum(dim=-1)  # Number of points above 0.95: batchSize, numSignals
        averageDiff = biomarkerDiff.mean(dim=-1) < self.maxAverageDiff  # Average difference between consecutive points: batchSize, numSignals
        minPointsMask = self.minSequencePoints <= allNumSignalPoints  # Minimum number of points: batchSize, numSignals
        validSignalMask = validDataMask.any(dim=-1)  # Missing data: batchSize, numSignals

        # Combine all masks into a single mask and expand to match dimensions.
        validSignalMask = minPointsMask & minLowerBoundaryMask & minUpperBoundaryMask & averageDiff & validSignalMask & singlePointMaxDiff
        validSignalInds = self.minSignalPresentCount < validSignalMask.sum(dim=0)

        # Filter out the invalid signals
        allSignalData[~validSignalMask.unsqueeze(-1).unsqueeze(-1).expand(batchSize, numSignals, maxSequenceLength, numChannels)] = 0
        allNumSignalPoints[~validSignalMask] = 0

        return allSignalData[:, validSignalInds, :, :], allNumSignalPoints[:, validSignalInds], featureNames[validSignalInds]

    def getSignalIntervals(self, experimentalData, eachSignal_numPoints, timeWindow, channelInds):
        # signalChannel: A torch array of size (numSignals, maxSequenceLength, [timeChannel, signalChannel])
        # eachSignal_numPoints: A torch array of size (numSignals)
        assert isinstance(channelInds, list), f"Expected a list of channel indices, but got {type(channelInds)}"

        experimentalIntervalData = []
        # For each signal in the batch.
        for signalInd in range(len(experimentalData)):
            signalData = experimentalData[signalInd]  # Dim: maxSequenceLength, [timeChannel, signalChannel]
            numSignalPoints = eachSignal_numPoints[signalInd]

            # Get the channel data.
            channelData = signalData[0:numSignalPoints, channelInds]
            channelTimes = signalData[0:numSignalPoints, -1]
            # channelData dim: maxSequenceLength, numChannels
            # channelTimes dim: maxSequenceLength

            # Get the signal interval.
            startSignalInd = self.dataAugmentation.getTimeIntervalInd(channelTimes, timeWindow, mustIncludeTimePoint=False)
            timeInterval = channelData[startSignalInd:numSignalPoints, :]

            # Store the interval information.
            experimentalIntervalData.append(timeInterval)

        return experimentalIntervalData

    @staticmethod
    def calculate_snr(biomarkerData, epsilon=1e-10):
        # biomarkerData dimension: numExperiments, numSignals, maxSequenceLength
        # Calculate the signal power and noise power for the current signal.
        signal_power = torch.mean(biomarkerData.pow(2), dim=-1)  # Signal power
        noise_power = torch.var(biomarkerData, dim=-1)  # Noise power (variance)

        # Calculate the SNR (adding epsilon to avoid log(0))
        snr_values = 10 * torch.log10((signal_power + epsilon) / (noise_power + epsilon))
        # snr_values dimension: numExperiments, numSignals

        return snr_values

    def normalizeSignals(self, allSignalData, missingDataMask):
        # signalBatchData dimension: numExperiments, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # missingDataMask dimension: numExperiments, numSignals, maxSequenceLength
        signalChannelInd = emotionDataInterface.getChannelInd(channelName=modelConstants.signalChannel)
        allSignalData[:, :, :, signalChannelInd] = self.generalMethods.minMaxScale_noInverse(allSignalData[:, :, :, signalChannelInd], scale=modelConstants.minMaxScale, missingDataMask=missingDataMask)

        return allSignalData
    