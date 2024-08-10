import numpy as np
import itertools
import pickle
import random
import torch
import gzip
import os

# Import helper files.
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from ..modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from .dataPreparation import minMaxScale_noInverse


class compileModelDataHelpers:
    def __init__(self, submodel, userInputParams, accelerator=None):
        # General parameters
        self.compiledInfoLocation = os.path.dirname(__file__) + "/../../../_experimentalData/_compiledData/"
        self.userInputParams = userInputParams
        self.missingLabelValue = torch.nan
        self.compiledExtension = ".pkl.gz"
        self.standardizeSignals = True
        self.accelerator = accelerator

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
        self.numIndices_perShift = None
        self.maxClassPercentage = None
        self.dontShiftDatasets = None
        self.numSecondsShift = None
        self.minNumClasses = None
        self.minSeqLength = None
        self.maxSeqLength = None
        self.numShifts = None

        # Set the submodel-specific parameters
        if submodel is not None: self.addSubmodelParameters(submodel, userInputParams)

    def addSubmodelParameters(self, submodel, userInputParams):
        if userInputParams is not None:
            self.userInputParams = userInputParams

        # Exclusion criterion.
        self.minNumClasses, self.maxClassPercentage = self.modelParameters.getExclusionCriteria(submodel)
        self.minSeqLength, self.maxSeqLength = self.modelParameters.getSequenceLengthRange(submodel, userInputParams['sequenceLength'])  # Seconds.

        # Data augmentation.
        samplingFrequency = 1
        self.dontShiftDatasets, self.numSecondsShift, numSeconds_perShift = self.modelParameters.getShiftInfo(submodel)
        # Data augmentation parameters calculations
        self.numShifts = 1 + self.numSecondsShift // numSeconds_perShift  # The first shift is the identity transformation.
        self.numIndices_perShift = (samplingFrequency * numSeconds_perShift)  # This must be an integer
        assert samplingFrequency == 1, "Check your code if samplingFrequency != 1 is okay."

        # Embedded information for each model.
        self.signalEncoderModelInfo = f"signalEncoder on {userInputParams['deviceListed']} with {userInputParams['signalEncoderWaveletType'].replace('.', '')} at {userInputParams['optimizerType']} at numSigLiftedChannels {userInputParams['numSigLiftedChannels']} at numExpandedSignals {userInputParams['numExpandedSignals']} at numSigEncodingLayers {userInputParams['numSigEncodingLayers']}"
        self.autoencoderModelInfo = f"autoencoder on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} at compressionFactor {str(userInputParams['compressionFactor']).replace('.', '')} expansionFactor {str(userInputParams['expansionFactor']).replace('.', '')}"
        self.emotionPredictionModelInfo = f"emotionPrediction on {userInputParams['deviceListed']} with {userInputParams['optimizerType']} with seqLength {userInputParams['sequenceLength']}"

    # ---------------------- Model Specific Parameters --------------------- #

    def embedInformation(self, submodel, trainingDate):
        if submodel == "signalEncoder":
            trainingDate = f"{trainingDate} {self.signalEncoderModelInfo}"
        elif submodel == "autoencoder":
            trainingDate = f"{trainingDate} {self.autoencoderModelInfo}"
        elif submodel == "emotionPrediction":
            trainingDate = f"{trainingDate} {self.emotionPredictionModelInfo}"
        else:
            raise Exception()
        print("trainingDate:", trainingDate, flush=True)

        return trainingDate

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
        """
        Purpose: To remove any activityNames where the label is not present and 
                 reindex the activityLabels from 0 to number of unique activities after culling.
        Parameters:
        - activityNames: a list of all unique activity names (strings).
        - activityLabels: a 1D tensor of index hashes for the unique activityName.
        """
        # Convert activityNames to a numpy array for easier string handling
        activityNames = np.asarray(activityNames)

        # Find the unique activity labels
        uniqueActivityLabels, culledActivityLabels = torch.unique(activityLabels, return_inverse=True)

        # Get the corresponding unique activity names
        uniqueActivityNames = activityNames[uniqueActivityLabels.int()]

        assert len(activityLabels) == len(culledActivityLabels)
        return uniqueActivityNames, culledActivityLabels.double()

    @staticmethod
    def segmentSignals_toModel(numSignals, numSignalsCombine, random_state, metaTraining=True):
        random.seed(random_state)

        # If it's our data.
        if not metaTraining:
            signalCombinationInds = itertools.combinations(range(0, numSignals), numSignalsCombine)
            assert numSignalsCombine <= numSignals, f"You cannot make {numSignalsCombine} combination with only {numSignals} signals."
        else:
            signalCombinationInds = []
            allCombinations = list(range(numSignals))
            for _ in range(1):
                # Create a list of all possible combinations of indices
                random.shuffle(allCombinations)

                # Iterate over the shuffled list and create signalCombinationInds
                for groupInd in range(0, len(allCombinations), numSignalsCombine):
                    # Form a group of random signals
                    group = allCombinations[groupInd:groupInd + numSignalsCombine]
                    # If the group is too small, add some more.
                    if len(group) < numSignalsCombine:
                        group.extend(allCombinations[0:numSignalsCombine - len(group)])

                    # Organize the signal groupings.
                    signalCombinationInds.append(group)

        print(f"\tPotentially Initializing {len(signalCombinationInds)} models per emotion", flush=True)
        return signalCombinationInds

    def organizeSignals(self, allSignalData):
        # allSignalData : A list of size (batchSize, numSignals, maxSequenceLength, 2)

        featureData = []
        # Compile the feature's data across all experiments.
        for batchInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[batchInd])

            # Assertions about data integrity.
            numSignals, sequenceLength = signalData.shape
            assert self.minSeqLength <= sequenceLength, f"Expected {self.minSeqLength}, but received {signalData.shape[1]} "

            # Standardize the signals
            if self.standardizeSignals:
                signalData = minMaxScale_noInverse(signalData, scale=self.modelParameters.getSignalMinMaxScale())

            # Add buffer if needed.
            if sequenceLength < self.maxSeqLength + self.numSecondsShift:
                prependedBuffer = np.zeros((numSignals, self.maxSeqLength + self.numSecondsShift - sequenceLength)) * signalData[:, 0:1]
                signalData = np.hstack((prependedBuffer, signalData))
            elif self.maxSeqLength + self.numSecondsShift < sequenceLength:
                signalData = signalData[:, -self.maxSeqLength - self.numSecondsShift:]

            # Standardize the signals
            if self.standardizeSignals:
                signalData = minMaxScale_noInverse(signalData, scale=self.modelParameters.getSignalMinMaxScale())

            # This is good signalData
            featureData.append(signalData.tolist())
        featureData = torch.tensor(featureData)

        # Assert the integrity.
        assert len(featureData) != 0

        return featureData

    def organizeLabels(self, allFeatureLabels, metaTraining, metaDatasetName, numSignals):
        # Convert to tensor and zero out the class indices
        allFeatureLabels = torch.as_tensor(allFeatureLabels)
        batchSize, numLabels = allFeatureLabels.shape

        allSingleClassIndices = []
        # For each type of label (emotion).
        for labelTypeInd in range(numLabels):
            # Get all the label responses across all batches.
            featureLabels = allFeatureLabels[:, labelTypeInd]
            allSingleClassIndices.append([])

            # Remove unknown labels.
            goodLabelInds = 0 <= featureLabels  # The minimum label should be 0
            featureLabels[~goodLabelInds] = self.missingLabelValue

            # Count the number of times the emotion label has a unique value.
            unique_classes, class_counts = torch.unique(featureLabels[goodLabelInds], return_counts=True)
            # unique_classes, class_counts dim: numUniqueClasses (number of unique emotion scores).
            smallClassInds = class_counts < 2

            # For small distributions.
            if smallClassInds.any():
                # Find the bad experiments (batches) with only one sample.
                badClasses = unique_classes[torch.nonzero(smallClassInds).reshape(1, -1)[0]]

                # Remove the datapoint as we cannot split the class between test/train
                smallClassLabelInds = torch.isin(featureLabels, badClasses)
                allSingleClassIndices[-1].extend(smallClassLabelInds)

                # Reassess the class counts
                finalMask = goodLabelInds & ~smallClassLabelInds
                unique_classes, class_counts = torch.unique(featureLabels[finalMask], return_counts=True)

            # Ensure greater variability in the class rating system.
            if metaTraining and (len(unique_classes) < self.minNumClasses or len(featureLabels) * self.maxClassPercentage <= class_counts.max().item()):
                featureLabels[:] = self.missingLabelValue

            # Save the edits made to the featureLabels
            allFeatureLabels[:, labelTypeInd] = featureLabels

        # Report the information from this dataset.
        numGoodEmotions = torch.sum(~torch.all(torch.isnan(allFeatureLabels), dim=0)).item()
        print(f"\t{metaDatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numLabels - 1}) well-labeled emotions across {batchSize} experiments with {numSignals} signals.", flush=True)

        return allFeatureLabels, allSingleClassIndices

    def addDemographicInfo(self, allFeatureData, allSubjectInds, datasetInd):
        """
        Purpose: The same signal without the last few seconds still has the same label
        --------------------------------------------
        allFeatureData : A 3D list of all signals in each experiment (batchSize, numSignals, sequenceLength)
        allSubjectInds : A 1D numpy array of size batchSize
        """
        # Get the dimensions of the input arrays
        numExperiments, numSignals, totalLength = allFeatureData.shape
        demographicLength = 0
        numSubjectIdentifiers = 2

        # Create lists to store the new augmented data.
        updatedFeatureData = torch.zeros((numExperiments, numSignals, self.maxSeqLength + demographicLength + 2))

        # For each recorded experiment.
        for experimentInd in range(numExperiments):
            # Compile an array of subject indices.
            subjectInds = torch.full((numSignals, 1), allSubjectInds[experimentInd])
            datasetInds = torch.full((numSignals, 1), datasetInd)

            # Collect the demographic information.
            demographicContext = torch.hstack((subjectInds, datasetInds))
            assert demographicLength + numSubjectIdentifiers == demographicContext.shape[1], "Asserting I am self-consistent. Hardcoded assertion"

            # Add the demographic data to the feature array.
            updatedFeatureData[experimentInd] = torch.hstack((allFeatureData[experimentInd], demographicContext))

        return updatedFeatureData, numSubjectIdentifiers, demographicLength

    # ---------------------------------------------------------------------- #
    # -------------------------- Data Augmentation ------------------------- #

    def addShiftedSignals(self, allFeatureData, allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds):
        """
        Purpose: The same signal without the last few seconds still has the same label
        --------------------------------------------
        allFeatureData : A 3D list of all signals in each experiment (batchSize, numSignals, sequenceLength)
        allFeatureLabels : A numpy array of all labels per experiment of size (batchSize, numLabels)
        currentTestingMask : A boolean mask of testing data of size (batchSize, numLabels)
        currentTrainingMask : A boolean mask of training data of size (batchSize, numLabels)
        allSubjectInds : A 1D numpy array of size batchSize
        """
        # Get the dimensions of the input arrays
        numExperiments, numSignals, totalLength = allFeatureData.shape
        _, numLabels = allFeatureLabels.shape

        # Create lists to store the new augmented data, labels, and masks
        augmentedFeatureLabels = torch.zeros((numExperiments * self.numShifts, numLabels))
        augmentedFeatureData = torch.zeros((numExperiments * self.numShifts, numSignals, self.maxSeqLength))
        augmentedTrainingMask = torch.full(augmentedFeatureLabels.shape, fill_value=False, dtype=torch.bool)
        augmentedTestingMask = torch.full(augmentedFeatureLabels.shape, fill_value=False, dtype=torch.bool)
        augmentedSubjectInds = torch.zeros((numExperiments * self.numShifts))

        # For each recorded experiment.
        for experimentInd in range(numExperiments):

            # For each shift in the signals.
            for shiftInd in range(self.numShifts):
                # Create shifted signals
                shiftedSignals = allFeatureData[experimentInd, :, -self.maxSeqLength - shiftInd * self.numIndices_perShift:totalLength - shiftInd * self.numIndices_perShift]

                # Append the shifted data and corresponding labels and masks
                augmentedFeatureLabels[experimentInd * self.numShifts + shiftInd] = allFeatureLabels[experimentInd]
                augmentedTrainingMask[experimentInd * self.numShifts + shiftInd] = currentTrainingMask[experimentInd]
                augmentedTestingMask[experimentInd * self.numShifts + shiftInd] = currentTestingMask[experimentInd]
                augmentedSubjectInds[experimentInd * self.numShifts + shiftInd] = allSubjectInds[experimentInd]
                augmentedFeatureData[experimentInd * self.numShifts + shiftInd] = shiftedSignals

        # import matplotlib.pyplot as plt
        # plt.plot(allFeatureData[0][0][-self.maxSeqLength:], 'k', linewidth=3, label="Original Curve")
        # plt.plot(augmentedFeatureData[0][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[1][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[2][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[3][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[4][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[5][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[6][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[7][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[8][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[9][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.xlabel("Time (Seconds)")
        # plt.ylabel("AU")
        # # plt.plot(augmentedFeatureData[0][0], 'tab:blue', linewidth=2)
        # plt.show()

        return augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds

    # ---------------------------------------------------------------------- #
    # ---------------------------- Data Cleaning --------------------------- #

    def _padSignalData(self, allRawFeatureTimeIntervals, allCompiledFeatureIntervals):
        # allRawFeatureTimeIntervals : A list of size (batchSize, numSignals, sequenceLength*)
        # allCompiledFeatureIntervals : A list of size (batchSize, numSignals, sequenceLength*)
        # allSignalData : A list of size (batchSize, numSignals, maxSequenceLength, 2)
        # allSignalStopInds : A list of size (batchSize, numSignals)            
        # Determine the final dimensions of the padded array.
        batchSize, numSignals = len(allRawFeatureTimeIntervals), len(allRawFeatureTimeIntervals[0])
        maxSequenceLength = max(max(len(seq) for seq in signal) for signal in allRawFeatureTimeIntervals)
        assert maxSequenceLength <= self.modelParameters.getMaxBufferLength(), f"{self.modelParameters.getMaxBufferLength()} < {maxSequenceLength}"

        # Initialize the padded array and end signal indices list
        allSignalData = np.zeros((batchSize, numSignals, maxSequenceLength, 2))
        allSignalStopInds = np.zeros(shape=(batchSize, numSignals), dtype=int)

        for batchInd in range(batchSize):
            for signalInd in range(numSignals):
                compiledFeatureInterval = allCompiledFeatureIntervals[batchInd][signalInd]
                rawTimeInterval = allRawFeatureTimeIntervals[batchInd][signalInd]

                # Determine the length of the sequence
                sequenceLength = len(compiledFeatureInterval)
                allSignalStopInds[batchInd][signalInd] = sequenceLength

                # Fill the padded array with the signal data
                allSignalData[batchInd][signalInd][:sequenceLength, 0] = rawTimeInterval
                allSignalData[batchInd][signalInd][:sequenceLength, 1] = compiledFeatureInterval

        return allSignalData, allSignalStopInds

    def _removeBadExperiments(self, allSignalData, allSignalStopInds, allLabels, subjectInds):
        """
        Purpose: Remove bad experiments from the data list.
        --------------------------------------------
        allSignalData : A numpy array of size (batchSize, numSignals, maxSequenceLength, 2)
        allSignalStopInds : A numpy array of size (batchSize, numSignals)
        allLabels : A numpy array of size (batchSize, numLabels)
        subjectInds : A numpy array of size (batchSize,)
        """
        # Initialize data holders.
        goodBatchInds = []

        # For each experimental batch
        for batchInd in range(len(allSignalStopInds)):
            minSignalInd = np.argmin(allSignalStopInds[batchInd])
            minBatchSequencePoints = allSignalStopInds[batchInd][minSignalInd]
            minBatchSequenceTime = allSignalData[batchInd][minSignalInd][minBatchSequencePoints - 1][0] - allSignalData[batchInd][minSignalInd][0][0]

            # Assert that the signal is long enough.
            if minBatchSequencePoints/minBatchSequenceTime < self.modelParameters.minFeatureFreq:
                print(f"\tBatch {batchInd} has a signal that is too short. {minBatchSequencePoints} points in {minBatchSequenceTime} seconds.")
                continue

            # Compile the good batches.
            goodBatchInds.append(batchInd)

        return allSignalData[goodBatchInds], allSignalStopInds[goodBatchInds], allLabels[goodBatchInds], subjectInds[goodBatchInds]

    def _removeBadSignals(self, allSignalData, allSignalStopInds, featureNames):
        """
        Purpose: Remove poor signals from ALL data batches.
        --------------------------------------------
        allSignalData : A numpy array of size (batchSize, numSignals, maxSequenceLength, 2)
        allSignalStopInds : A numpy array of size (batchSize, numSignals)
        featureNames : A numpy array of size numSignals
        """

        # Ensure the feature names match the number of signals
        assert len(allSignalData) == 0 or len(featureNames) == allSignalData.shape[1], \
            f"Feature names do not match data dimensions. {len(featureNames)} != {allSignalData.shape[1]}"

        # Initialize a mask for good features (start with all features assumed good)
        goodFeatureMask = np.ones(len(featureNames), dtype=bool)

        # For each batch of signals
        for batchInd in range(allSignalData.shape[0]):
            signalBatchData = allSignalData[batchInd]  # shape: (numSignals, maxSequenceLength, 2)
            signalStopInds = allSignalStopInds[batchInd]

            # Calculate SNRs for each signal in the batch
            signalSNRs = self.calculate_snr(signalBatchData, signalStopInds)

            # Update the good feature mask based on an SNR threshold
            goodFeatureMask &= (1E-10 <= signalSNRs)

        # Apply the mask to keep only the good signals and corresponding feature names
        return allSignalData[:, goodFeatureMask], allSignalStopInds[:, goodFeatureMask], featureNames[goodFeatureMask]

    @staticmethod
    def calculate_snr(signalBatchData, signalStopInds):
        # signalBatchData dimension: numSignals, maxSequenceLength, 2
        # signalStopInds dimension: numSignals
        snr_values = np.zeros(len(signalBatchData))

        # For each signal in the batch.
        for signalInd in range(len(signalBatchData)):
            # Get the signal data for the current signal.
            signalData = signalBatchData[signalInd, :signalStopInds[signalInd], 1]

            # Standardize the signals (min-max scaling).
            scaledData = minMaxScale_noInverse(signalData[:, :, 1], scale=1)

            # Calculate the signal-to-noise ratio for each signal.
            signal_power = np.mean(scaledData ** 2, axis=-1)
            noise_power = np.var(scaledData, axis=-1)

            # Handle the case when noise_power == 0 (avoid division by zero)
            signal_power[noise_power == 0] = 1E-10  # Prevent division by zero
            noise_power[noise_power == 0] = 1E-10  # Prevent division by zero

            # Calculate the signal-to-noise ratio for each signal.
            snr_values[signalInd] = 10 * np.log10(signal_power / noise_power)

        return snr_values

    # ---------------------------------------------------------------------- #
