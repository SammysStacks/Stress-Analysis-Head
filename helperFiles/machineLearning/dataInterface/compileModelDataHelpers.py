import gzip
import os
import pickle
from bisect import bisect_right

import numpy as np
import torch

from .dataPreparation import minMaxScale_noInverse
from ..modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
# Import helper files.
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters


class compileModelDataHelpers:
    def __init__(self, submodel, userInputParams, accelerator=None):
        # General parameters
        self.compiledInfoLocation = os.path.dirname(__file__) + "/../../../_experimentalData/_compiledData/"
        self.userInputParams = userInputParams
        self.missingLabelValue = torch.nan
        self.compiledExtension = ".pkl.gz"
        self.standardizeSignals = True
        self.accelerator = accelerator
        self.verbose = False

        # Make Output Folder Directory if Not Already Created
        os.makedirs(self.compiledInfoLocation, exist_ok=True)

        # Initialize relevant classes.
        self.modelParameters = modelParameters(userInputParams,  accelerator)
        self.modelMigration = modelMigration(accelerator, debugFlag=False)
        self.dataInterface = emotionDataInterface

        # Submodel-specific parameters
        self.emotionPredictionModelInfo = None
        self.signalEncoderModelInfo = None
        self.autoencoderModelInfo = None
        self.maxClassPercentage = None
        self.minNumClasses = None

        # Data cleaning parameters
        self.minSequencePoints_perSmallestTimeWindow = modelConstants.timeWindows[0] / 10
        self.maxTimeGap_perLargestTimeWindow = 60

        # Set the submodel-specific parameters
        if submodel is not None: self.addSubmodelParameters(submodel, userInputParams)

    def addSubmodelParameters(self, submodel, userInputParams):
        if userInputParams is not None:
            self.userInputParams = userInputParams

        # Exclusion criterion.
        self.minNumClasses, self.maxClassPercentage = self.modelParameters.getExclusionCriteria(submodel)

        # Embedded information for each model.
        self.signalEncoderModelInfo = f"signalEncoder on {userInputParams['deviceListed']} with {userInputParams['signalEncoderWaveletType'].replace('.', '')} at {userInputParams['optimizerType']} at numSigLiftedChannels {userInputParams['numSigLiftedChannels']} at numExpandedSignals {userInputParams['numExpandedSignals']} at numSigEncodingLayers {userInputParams['numSigEncodingLayers']}"
        self.autoencoderModelInfo = f"autoencoder on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} at compressionFactor {str(userInputParams['compressionFactor']).replace('.', '')} expansionFactor {str(userInputParams['expansionFactor']).replace('.', '')}"
        self.emotionPredictionModelInfo = f"emotionPrediction on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} with seqLength {userInputParams['finalDistributionLength']}"

    # ---------------------- Model Specific Parameters --------------------- #

    def embedInformation(self, submodel, trainingDate):
        if submodel == modelConstants.signalEncoderModel:
            trainingDate = f"{trainingDate} {self.signalEncoderModelInfo}"
        elif submodel == modelConstants.autoencoderModel:
            trainingDate = f"{trainingDate} {self.autoencoderModelInfo}"
        elif submodel == modelConstants.emotionPredictionModel:
            trainingDate = f"{trainingDate} {self.emotionPredictionModelInfo}"
        else:
            raise Exception()
        print("trainingDate:", trainingDate, flush=True)

        return trainingDate

    # ---------------------- Saving/Loading Model Data --------------------- #
    def saveCompiledInfo(self, data_to_store, saveDataName):
        with gzip.open(filename=f'{self.compiledInfoLocation}{saveDataName}{self.compiledExtension}', mode='wb') as file:
            file.write(data_to_store.encode("utf-8"))  # Encode string to bytes

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
        uniqueActivityNames = np.asarray(activityNames)[uniqueActivityLabels.int()]

        return uniqueActivityNames, validActivityLabels.to(torch.int)

    def organizeLabels(self, allFeatureLabels, metaTraining, metaDatasetName, numSignals):
        # allFeatureLabels: A numpy array or list of size (batchSize, numLabels)
        # metaTraining: Boolean indicating if the data is for training
        # metaDatasetName: String representing the name of the dataset
        # numSignals: The number of signals in the dataset
        # Convert to tensor and initialize lists
        allFeatureLabels = torch.as_tensor(allFeatureLabels)
        batchSize, numLabels = allFeatureLabels.shape
        allSingleClassIndices = [[]]*numLabels

        # Iterate over each label type (emotion)
        for labelTypeInd in range(numLabels):
            featureLabels = allFeatureLabels[:, labelTypeInd]

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
                allSingleClassIndices[labelTypeInd].extend(smallClassLabelMask.cpu().numpy().tolist())
                featureLabels[smallClassLabelMask] = self.missingLabelValue

                # Recalculate unique classes.
                goodLabelInds = goodLabelInds & ~smallClassLabelMask
                unique_classes, class_counts = torch.unique(featureLabels[goodLabelInds], return_counts=True)

            # Ensure greater variability in the class rating system.
            if metaTraining and (len(unique_classes) < self.minNumClasses or len(featureLabels) * self.maxClassPercentage <= class_counts.max().item()):
                featureLabels[:] = self.missingLabelValue

            # Save the edits made to the featureLabels
            allFeatureLabels[:, labelTypeInd] = featureLabels

        # Report the information from this dataset.
        numGoodEmotions = torch.sum(~torch.all(torch.isnan(allFeatureLabels), dim=0)).item()
        print(f"\t{metaDatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numLabels - 1}) well-labeled emotions across {batchSize} experiments with {numSignals} signals.", flush=True)

        return allFeatureLabels, allSingleClassIndices

    @staticmethod
    def addDemographicInfo(allSignalData, allNumSignalPoints, allSubjectInds, datasetInd):
        # allSignalData: A numpy array of size (batchSize, numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward, time])
        # allNumSignalPoints: A numpy array of size (batchSize, numSignals)
        # allSubjectInds: A numpy array of size batchSize
        numExperiments, numSignals, maxSequenceLength, numChannels = allSignalData.shape
        assert len(modelConstants.signalChannelNames) == numChannels - 1
        numSubjectIdentifiers = len(modelConstants.subjectIdentifiers)

        # Create lists to store the new augmented data.
        compiledSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength + numSubjectIdentifiers, numChannels))

        # For each recorded experiment.
        for experimentInd in range(numExperiments):
            # Compile an array of subject indices.
            subjectInds = torch.full(size=(numSignals, 1, numChannels), fill_value=allSubjectInds[experimentInd])
            datasetInds = torch.full(size=(numSignals, 1, numChannels), fill_value=datasetInd)

            # Compile an array of signal stop indices.
            eachSignal_numPoints = allNumSignalPoints[experimentInd].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, numChannels)

            # Collect the demographic information.
            demographicContext = torch.hstack((eachSignal_numPoints, datasetInds, subjectInds))
            assert numSubjectIdentifiers == demographicContext.shape[1], "Asserting I am self-consistent. Hardcoded assertion"

            # Add the demographic data to the feature array.
            compiledSignalData[experimentInd] = torch.hstack((allSignalData[experimentInd], demographicContext))

        return compiledSignalData

    # ---------------------------------------------------------------------- #
    # ---------------------------- Data Cleaning --------------------------- #

    @staticmethod
    def _padSignalData(allRawFeatureTimeIntervals, allCompiledFeatureIntervals):
        # allCompiledFeatureIntervals : batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
        # allRawFeatureTimeIntervals : batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant
        # allSignalData : A list of size (batchSize, numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward, time])
        # allNumSignalPoints : A list of size (batchSize, numSignals)            
        # Determine the final dimensions of the padded array.
        maxSequenceLength = max(max(len(biomarkerData) for biomarkerData in experimentalData) for experimentalData in allCompiledFeatureIntervals)
        numSignals = sum(len(biomarkerData[0]) for biomarkerData in allCompiledFeatureIntervals[0])
        numExperiments = len(allCompiledFeatureIntervals)

        # Initialize the padded array and end signal indices list
        allSignalData = torch.zeros(size=(numExperiments, numSignals, maxSequenceLength, len(modelConstants.signalChannelNames) + 1), dtype=torch.float32)  # +1 for the time data
        allNumSignalPoints = torch.zeros(size=(numExperiments, numSignals), dtype=torch.int)

        # For each batch of biomarkers.
        for experimentalInd in range(numExperiments):
            batchData = allCompiledFeatureIntervals[experimentalInd]
            batchTimes = allRawFeatureTimeIntervals[experimentalInd]

            currentSignalInd = 0
            # For each biomarker in the batch.
            for biomarkerInd in range(len(batchData)):
                biomarkerData = torch.as_tensor(batchData[biomarkerInd]).T  # Dim: numBiomarkerFeatures, batchSpecificFeatureLength
                biomarkerTimes = torch.as_tensor(batchTimes[biomarkerInd])  # Dim: batchSpecificFeatureLength

                # Get the number of signals in the current biomarker.
                numBiomarkerFeatures, batchSpecificFeatureLength = biomarkerData.shape
                finalSignalInd = currentSignalInd + numBiomarkerFeatures

                # Get the time differences.
                timeGaps = torch.diff(biomarkerTimes, n=1, dim=-1)
                # The time gap between point n and n+1  ->  f(n+1)-f(n)

                # Fill the padded array with the signal data
                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0:batchSpecificFeatureLength, 0] = biomarkerData
                # allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 1: batchSpecificFeatureLength, 1] = biomarkerData[:, 0:-1]
                # allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0: batchSpecificFeatureLength-1, 2] = biomarkerData[:, 1:batchSpecificFeatureLength]
                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 1:batchSpecificFeatureLength, 1] = timeGaps
                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0:batchSpecificFeatureLength-1, 2] = timeGaps
                allSignalData[experimentalInd, currentSignalInd:finalSignalInd, 0:batchSpecificFeatureLength, 3] = biomarkerTimes[-1] - biomarkerTimes
                allNumSignalPoints[experimentalInd, currentSignalInd:finalSignalInd] = batchSpecificFeatureLength

                # Update the current signal index
                currentSignalInd = finalSignalInd

        return allSignalData, allNumSignalPoints

    def getSignalIntervals(self, experimentalData, eachSignal_numPoints, timeWindow, channelInds):
        # signalData: A numpy array of size (numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward, time])
        # eachSignal_numPoints: A numpy array of size (numSignals)
        assert isinstance(channelInds, list), f"Expected a list of channel indices, but got {type(channelInds)}"

        experimentalIntervalData = []
        # For each signal in the batch.
        for signalInd in range(len(experimentalData)):
            signalData = experimentalData[signalInd]  # Dim: maxSequenceLength, [signal, dTimeBack, dTimeForward, time]
            numSignalPoints = eachSignal_numPoints[signalInd]

            # Get the channel data.
            channelData = signalData[0:numSignalPoints, channelInds]
            channelTimes = signalData[0:numSignalPoints, -1]
            # channelData dim: maxSequenceLength, numChannels
            # channelTimes dim: maxSequenceLength

            # Get the signal interval.
            startSignalInd = self.getTimeIntervalInd(channelTimes, timeWindow, mustIncludeTimePoint=False)
            timeInterval = channelData[startSignalInd:numSignalPoints, :]

            # Store the interval information.
            experimentalIntervalData.append(timeInterval)

        return experimentalIntervalData

    @staticmethod
    def getTimeIntervalInd(timeData, timePoint, mustIncludeTimePoint=False):
        # Assert the validity of the input parameters.
        assert 0 <= timePoint, f"Expected a positive time point, but got {timePoint}"
        timeData = torch.as_tensor(timeData)  # Ensure timeData is a torch tensor

        # Find the index of the time point in the time data
        timeInd = torch.where(timePoint <= timeData)[0][0].item()
        isTimePointIncluded = timeData[0] <= timePoint

        # Include the time point if necessary
        if not isTimePointIncluded and mustIncludeTimePoint:
            timeInd = max(timeInd - 1, 0)

        return timeInd

    def _removeBadExperiments(self, allSignalData, allNumSignalPoints, allLabels, subjectInds):
        """
        Purpose: Remove bad experiments from the data list.
        --------------------------------------------
        allSignalData : A numpy array of size (batchSize, numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward, time])
        allNumSignalPoints : A numpy array of size (batchSize, numSignals)
        allLabels : A numpy array of size (batchSize, numLabels)
        subjectInds : A numpy array of size (batchSize,)
        """
        # Initialize data holders.
        batchSize, numSignals, maxSequenceLength, numChannels = allSignalData.shape

        validExperimentalInds = []
        # For each experimental batch
        for experimentalInd in range(batchSize):
            eachSignal_numPoints = allNumSignalPoints[experimentalInd]
            experimentalData = allSignalData[experimentalInd]

            # Get the time intervals of the signals.
            maxTimeIntervalData = self.getSignalIntervals(experimentalData, eachSignal_numPoints, modelConstants.timeWindows[-1], channelInds=[-1])
            minTimeIntervalData = self.getSignalIntervals(experimentalData, eachSignal_numPoints, modelConstants.timeWindows[0], channelInds=[-1])
            # minSignalIntervalData, maxSignalIntervalData Dim: numSignals, *numIntervalPoints  ->  *numIntervalPoints is not constant

            # Calculate the longest time gap within the longest time window.
            maxTimeGap = max((max(torch.diff(timeIntervalData, n=1, dim=-1, prepend=timeIntervalData[0])) for timeIntervalData in maxTimeIntervalData))

            # Remove the batch if the gap is large.
            if self.maxTimeGap_perLargestTimeWindow < maxTimeGap:
                if self.verbose: print(f"\t\tBatch {experimentalInd} has a time gap of {maxTimeGap} which is too large. Removing this batch.", flush=True)
                continue

            # Calculate the number of points within the smallest time window.
            numMinWindowPoints = min(len(timeIntervalData) for timeIntervalData in minTimeIntervalData)

            # Remove the batch if the signal is too short.
            if numMinWindowPoints < self.minSequencePoints_perSmallestTimeWindow:
                if self.verbose: print(f"\t\tBatch {experimentalInd} has a signal that is too short ({numMinWindowPoints}). Removing this batch.", flush=True)
                continue

            # Compile the good batches.
            validExperimentalInds.append(experimentalInd)

        # Convert to tensor arrays for easier handling
        allNumSignalPoints = torch.as_tensor(allNumSignalPoints, dtype=torch.float32)[validExperimentalInds]
        allSignalData = torch.as_tensor(allSignalData, dtype=torch.float32)[validExperimentalInds]
        subjectInds = torch.as_tensor(subjectInds, dtype=torch.int)[validExperimentalInds]
        allLabels = torch.as_tensor(allLabels, dtype=torch.float32)[validExperimentalInds]
        # allSignalData dim: numValidExperiments, numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward, time]
        # allNumSignalPoints dim: numValidExperiments, numSignals
        # allLabels dim: numValidExperiments, numLabels
        # subjectInds dim: numValidExperiments

        return allSignalData, allNumSignalPoints, allLabels, subjectInds

    def _preprocessSignals(self, allSignalData, allNumSignalPoints, featureNames):
        # allSignalData: A numpy array of size (batchSize, numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward])
        # allNumSignalPoints: A numpy array of size (batchSize, numSignals)
        # featureNames: A numpy array of size numSignals
        # Ensure the feature names match the number of signals
        assert len(allSignalData) == 0 or len(featureNames) == allSignalData.shape[1], \
            f"Feature names do not match data dimensions. {len(featureNames)} != {allSignalData.shape[1]}"

        # Initialize a mask for good features (start with all features assumed good)
        validSignalInds = np.ones(len(featureNames), dtype=bool)

        # For each batch of signals
        for experimentalInd in range(allSignalData.shape[0]):
            eachSignal_numPoints = allNumSignalPoints[experimentalInd]

            # Standardize the signals.
            allSignalData[experimentalInd] = self.normalizeSignals(allSignalData[experimentalInd], eachSignal_numPoints)

            # Calculate SNRs for each signal in the batch
            signalSNRs = self.calculate_snr(allSignalData[experimentalInd, :, :, 0], eachSignal_numPoints)

            # Update the good feature mask based on an SNR threshold
            validSignalInds &= (1E-10 <= signalSNRs)

        # Convert to tensor for easier handling
        allNumSignalPoints = torch.as_tensor(allNumSignalPoints[:, validSignalInds], dtype=torch.int)
        allSignalData = torch.as_tensor(allSignalData[:, validSignalInds], dtype=torch.float32)

        # Apply the mask to keep only the good signals and corresponding feature names
        return allSignalData, allNumSignalPoints, featureNames[validSignalInds]

    @staticmethod
    def calculate_snr(signalBatchData, eachSignal_numPoints):
        # signalBatchData dimension: numSignals, maxSequenceLength
        # eachSignal_numPoints dimension: numSignals
        snr_values = np.zeros(len(signalBatchData))

        # For each signal in the batch.
        for signalInd in range(len(signalBatchData)):
            # Get the signal data for the current signal.
            signalData = signalBatchData[signalInd, 0:eachSignal_numPoints[signalInd]]  # Dim: maxSequenceLength

            # Calculate the signal-to-noise ratio for each signal.
            signal_power = np.mean(signalData ** 2, axis=-1)
            noise_power = np.var(signalData, axis=-1)

            # Calculate the signal-to-noise ratio for each signal.
            if signal_power == 0 or noise_power == 0: snr_values[signalInd] = 0
            else: snr_values[signalInd] = 10 * np.log10(signal_power / noise_power)

        return snr_values

    def normalizeSignals(self, signalBatchData, eachSignal_numPoints):
        # signalBatchData dimension: numSignals, maxSequenceLength, [signal, dTimeBack, dTimeForward]
        # eachSignal_numPoints dimension: numSignals
        # For each signal in the batch.
        for signalInd in range(len(signalBatchData)):
            numSignalPoints = eachSignal_numPoints[signalInd]

            # Standardize the signals (min-max scaling).
            signalBatchData[signalInd, 0:numSignalPoints, 0] = minMaxScale_noInverse(signalBatchData[signalInd, 0:numSignalPoints, 0], scale=modelConstants.minMaxScale)

            # Standardize the times (min-max scaling).
            signalBatchData[signalInd, 0:numSignalPoints, 1] /= self.maxTimeGap_perLargestTimeWindow
            signalBatchData[signalInd, 0:numSignalPoints, 2] /= self.maxTimeGap_perLargestTimeWindow

        return signalBatchData

    # ---------------------------------------------------------------------- #
