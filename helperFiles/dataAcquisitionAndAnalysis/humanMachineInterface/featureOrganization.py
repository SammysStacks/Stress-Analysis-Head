from scipy.interpolate import Akima1DInterpolator
from bisect import bisect_left, bisect_right
import collections
import numpy as np
import torch
import scipy

# Import files.
from .humanMachineInterface import humanMachineInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants

# Parameters for the streamingProtocolHelpers class:
#     Biomarker information:
#         biomarkerFeatureOrder: A list of biomarkers that require features in streamingOrder; Dim: numFeatureSignals
#         featureAnalysisOrder: A list of unique biomarkers that require features in streamingOrder; Dim: numUniqueFeatureSignals
#
#     Analysis information:
#         featureAnalysisList: A list of unique analysis protocols in featureAnalysisOrder; Dim: numUniqueFeatureSignals
#         analysisList: A list of unique analysis protocols; Dim: numUniqueSignals
#
#     Feature information:
#         rawFeatureTimesHolder: A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
#         rawFeatureHolder: A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures


class featureOrganization(humanMachineInterface):

    def __init__(self, modelClasses, actionControl, analysisProtocols, extractFeaturesFrom, featureAverageWindows):
        super().__init__(modelClasses, actionControl, extractFeaturesFrom)
        # General parameters.
        self.featureAnalysisOrder = list(collections.OrderedDict.fromkeys(self.biomarkerFeatureOrder))  # The set of unique feature biomarkers, maintaining the order they will be analyzed. Ex: ['eog', 'eeg', 'eda']
        self.newSamplingFreq = self.compileModelInfo.newSamplingFreq  # The new sampling frequency for the data for the machine learning model.
        self.minNumExperimentalPoints = 10  # The minimum number of points required for an experimental interval.
        self.biomarkerFeatureInds = []  # The indices of the biomarker features in the featureNames; Dim: numBiomarkers, numBiomarkerFeatures
        self.featureAnalysisList = []  # A list of unique analyses that will have features, keeping the order they are streamed in.
        self.interpBufferPoints = 8  # The buffer for the edge of the feature interpolation.
        self.trimMeanCut = 0.25  # The proportion to cut for the trimmed mean.
        self.analysisProtocols = analysisProtocols
        # Assert the integrity of feature organization.
        assert len(featureAverageWindows) == len(self.biomarkerFeatureOrder), f"Found {featureAverageWindows} windows for {self.biomarkerFeatureOrder} biomarkers. These must to be the same length."

        # Loop through each analysis requiring collecting features.

        for biomarkerInd in range(len(self.featureAnalysisOrder)):
            biomarkerType = self.featureAnalysisOrder[biomarkerInd]

            # Get the channel indices for the biomarker.
            featureChannelIndices = np.where(self.biomarkerFeatureOrder == biomarkerType)[0]

            # Specify the parameters to collect features.
            analysisProtocols[biomarkerType].setFeatureCollectionParams(featureAverageWindows[biomarkerInd], featureChannelIndices)
            self.featureAnalysisList.append(analysisProtocols[biomarkerType])

        startBiomarkerInd = 0; endBiomarkerInd = 0
        # Loop through each biomarker's list of features names.
        for biomarkerInd in range(len(self.biomarkerFeatureNames)):
            numBiomarkerFeatures = len(self.biomarkerFeatureNames[biomarkerInd])
            endBiomarkerInd = startBiomarkerInd + numBiomarkerFeatures

            # Store the feature indices for this biomarker.
            self.biomarkerFeatureInds.append(np.arange(startBiomarkerInd, endBiomarkerInd, 1))
            startBiomarkerInd = endBiomarkerInd
        assert endBiomarkerInd == len(self.featureNames), f"Found {endBiomarkerInd} biomarker features and {len(self.featureNames)} features. These must be the same length."

        self.therapyPointers = None
        # Initialize mutable variables.
        self.resetFeatureInformation()

    def resetFeatureInformation(self):
        self.resetVariables_HMI()
        # Reset the analysis information
        for featureAnalysis in self.featureAnalysisList:
            featureAnalysis.resetAnalysisVariables()
            featureAnalysis.resetGlobalVariables()

        # Raw feature data structure
        self.therapyPointers = [0 for _ in range(len(self.analysisProtocols))]  # A list of pointers indicating the last seen raw feature index for each analysis and each channel.
        self.rawFeatureTimesHolder = [[] for _ in range(len(self.biomarkerFeatureOrder))]  # A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
        self.rawFeatureHolder = [[] for _ in range(len(self.biomarkerFeatureOrder))]  # A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures

        # Feature collection parameters
        self.rawFeaturePointers = np.zeros(len(self.biomarkerFeatureOrder), dtype=int)  # A list of pointers indicating the last seen raw feature index for each analysis.

    def unifyFeatureTimeWindows(self, featureTimeWindow):
        for featureAnalysis in self.featureAnalysisList:
            featureAnalysis.featureTimeWindow = featureTimeWindow

    # --------------------- Organize Incoming Features --------------------- #

    def compileIntervalFeaturesWithPadding(self):
        # Find the number of new points.
        lastRecordedTime = self.featureAnalysisList[0].timepoints[-1]
        numNewPoints = int(1 + (lastRecordedTime - self.startModelTime - self.modelTimeBuffer) // self.modelTimeGap)

        modelTimes = []
        allRawFeatureTimeInterval = []
        allRawFeatureInterval = []

        # for each new point
        for numNewPoint in range(numNewPoints):
            endModelTime = self.startModelTime - self.modelTimeWindow

            pointFeatureTimes = []
            pointFeatures = []

            # For each unique analysis with features.
            for biomarkerInd in range(len(self.featureAnalysisList)):
                oldEndPointer = self.therapyPointers[biomarkerInd]

                biomarkerTimes = self.rawFeatureTimesHolder[biomarkerInd]
                biomarkerFeatures = self.rawFeatureHolder[biomarkerInd]

                # Locate the experiment indices within the data
                newStartTimePointer = oldEndPointer + np.searchsorted(self.rawFeatureTimesHolder[0][oldEndPointer:], self.startModelTime, side='right') + 1
                newEndTimePointer = oldEndPointer + np.searchsorted(self.rawFeatureTimesHolder[0][oldEndPointer:], endModelTime, side='left')
                numBiomarkerPoints = newStartTimePointer - newEndTimePointer
                self.therapyPointers[biomarkerInd] = newEndTimePointer

                # Extract the time and feature intervals
                biomarkerTimeInterval = biomarkerTimes[newEndTimePointer:newStartTimePointer]
                biomarkerFeaturesInterval = biomarkerFeatures[newEndTimePointer:newStartTimePointer]

                pointFeatureTimes.append(biomarkerTimeInterval)
                pointFeatures.append(biomarkerFeaturesInterval)

            allRawFeatureTimeInterval.append(pointFeatureTimes)
            allRawFeatureInterval.append(pointFeatures)

        startModelTimePerBiomarker = [[self.startModelTime] for _ in range(numNewPoints)]
        startModelTimePerBiomarker = torch.tensor(startModelTimePerBiomarker, dtype=torch.float32)
        allSignalData, allNumSignalPoints = self.compileModelHelpers._padSignalData(allRawFeatureTimeInterval, allRawFeatureInterval, startModelTimePerBiomarker)
        # Update the model time.
        self.startModelTime += self.modelTimeGap
        modelTimes.append(self.startModelTime)
        # Update the pointers.
        return modelTimes, allSignalData, allNumSignalPoints

    def organizeRawFeatures(self):
        # For each unique analysis with features.
        for analysis in self.featureAnalysisList:

            # For each channel in the analysis.
            for featureChannelInd in range(len(analysis.featureChannelIndices)):
                biomarkerInd = analysis.featureChannelIndices[featureChannelInd]
                rawFeaturePointer = self.rawFeaturePointers[biomarkerInd]

                # Organize the raw features; NOTE: I am assuming that the raw features are in order of the featureChannelIndices.
                self.rawFeatureTimesHolder[biomarkerInd].extend(analysis.rawFeatureTimes[featureChannelInd][rawFeaturePointer:])
                self.rawFeatureHolder[biomarkerInd].extend(analysis.rawFeatures[featureChannelInd][rawFeaturePointer:])

                # Update the raw pointers.
                self.rawFeaturePointers[biomarkerInd] = len(analysis.rawFeatureTimes[featureChannelInd])

                # Assert the integrity of the raw feature organization.
                assert len(self.rawFeatureTimesHolder[biomarkerInd]) == len(analysis.rawFeatureTimes[featureChannelInd]), \
                    f"Found {len(self.rawFeatureTimesHolder[biomarkerInd])} raw times and {len(self.rawFeatureHolder[biomarkerInd])} raw features. These must be the same length."
                assert len(self.rawFeatureHolder[biomarkerInd]) == len(analysis.rawFeatures[featureChannelInd]), \
                    f"Found {len(self.rawFeatureHolder[biomarkerInd])} raw features and {len(analysis.rawFeatures[featureChannelInd])} raw features. These must be the same length."

    def findCommonTimeRange(self):
        # Set up the parameters.
        rightAlignedTime = np.inf
        leftAlignedTime = 0

        # For each biomarker with features.
        for biomarkerInd in range(len(self.biomarkerFeatureOrder)):
            rawFeatureTimes = self.rawFeatureTimesHolder[biomarkerInd]

            # Update the min/max times.
            if 10 < len(rawFeatureTimes):  # Check if there are enough points to interpolate.
                leftAlignedTime = max(leftAlignedTime, rawFeatureTimes[0])
                rightAlignedTime = min(rightAlignedTime, rawFeatureTimes[-1])
            else:
                return None, None

        # Check if the time range is valid.
        if np.isinf(rightAlignedTime):
            return None, None

        return leftAlignedTime, rightAlignedTime

    # ---------------------- Compile Incoming Features --------------------- #

    def compileModelFeatures(self, startTime, endTime, featureTimes, features):
        # features dim: numTimePoints, numBiomarkerFeatures
        # featureTimes dim: numTimePoints

        # Locate the experiment indices within the data
        startStimuliInd = np.searchsorted(featureTimes, startTime, side='right')
        endStimuliInd = np.searchsorted(featureTimes, endTime, side='left')

        # Check if the time range is valid.
        if endStimuliInd - startStimuliInd <= self.minNumExperimentalPoints:
            return None, None

        # Save raw interval information
        featureIntervalTimes = featureTimes[startStimuliInd:endStimuliInd]
        featureIntervals = features[startStimuliInd:endStimuliInd]

        return featureIntervals, featureIntervalTimes

    def averageFeatures_static(self, rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=0):
        # rawFeatures dim: numTimePoints, numBiomarkerFeatures
        # rawFeatureTimes dim: numTimePoints
        minStartInd = bisect_right(rawFeatureTimes, rawFeatureTimes[0] + averageWindow + 1)
        startTimeInd = max(startTimeInd, minStartInd)

        compiledFeatures, compiledFeatureTimes = [], []
        # Average the Feature Together at Each Point
        for timePointInd in range(startTimeInd, len(rawFeatureTimes)):
            currentTimepoint = rawFeatureTimes[timePointInd]

            # Get the interval of features to average
            windowTimeInd = bisect_left(rawFeatureTimes, currentTimepoint - averageWindow)
            featureInterval = rawFeatures[windowTimeInd:timePointInd + 1]
            # Take the trimmed average
            compiledFeature = scipy.stats.trim_mean(featureInterval, proportiontocut=self.trimMeanCut, axis=0).tolist()
            compiledFeatureTimes.append(currentTimepoint)
            compiledFeatures.append(compiledFeature)
        # compiledFeatures dim: numTimePoints, numBiomarkerFeatures
        # compiledFeatureTimes dim: numTimePoints

        return compiledFeatureTimes, compiledFeatures

    def compileContinuousFeatures(self, newFeatureTimes, newRawFeatures, rawFeatureTimes, rawFeatures, compiledFeatureTimes, compiledFeatures, averageWindow):
        # newRawFeatures dim: numNewTimePoints, numBiomarkerFeatures
        # compiledFeatures dim: numTimePoints, numBiomarkerFeatures
        # rawFeatures dim: numTimePoints, numBiomarkerFeatures
        # newFeatureTimes dim: numNewTimePoints
        # rawFeatureTimes dim: numTimePoints

        # Assert the integrity of the feature compilation.
        assert len(newFeatureTimes) == len(newRawFeatures), f"Found {len(newFeatureTimes)} new times and {len(newRawFeatures)} new features. These must be the same length."
        assert len(rawFeatureTimes) == len(rawFeatures), f"Found {len(rawFeatureTimes)} times and {len(rawFeatures)} features. These must be the same length."
        if len(newFeatureTimes) == 0: return None

        startTimeInd = len(rawFeatures)
        # Append the new features to the raw features
        rawFeatureTimes.extend(newFeatureTimes)
        rawFeatures.extend(newRawFeatures)

        # Perform the feature averaging
        newCompiledFeatureTimes, newCompiledFeatures = self.averageFeatures_static(rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=startTimeInd)
        compiledFeatureTimes.extend(newCompiledFeatureTimes)
        compiledFeatures.extend(newCompiledFeatures)

        # Assert the integrity of the feature compilation.
        assert len(rawFeatures) == len(rawFeatureTimes), f''
        assert len(compiledFeatures) == len(compiledFeatureTimes), f''

    def compileStaticFeatures(self, rawFeatureTimesHolder, rawFeatureHolder, featureAverageWindows):
        # rawFeatureHolder dim: numBiomarkers, numTimePoints, numBiomarkerFeatures
        # rawFeatureTimesHolder dim: numBiomarkers, numTimePoints
        # featureAverageWindows dim: numBiomarkers

        # Assert the integrity of the feature compilation.
        assert len(rawFeatureTimesHolder) == len(rawFeatureHolder), \
            f"Found {len(rawFeatureTimesHolder)} times and {len(rawFeatureHolder)} features. These must be the same length."

        compiledFeatureHolders, compiledFeatureTimesHolders = [], []
        # Average the features across a sliding window at each timePoint
        for biomarkerInd in range(len(rawFeatureTimesHolder)):
            rawFeatureTimes = rawFeatureTimesHolder[biomarkerInd]
            averageWindow = featureAverageWindows[biomarkerInd]
            rawFeatures = rawFeatureHolder[biomarkerInd]

            # Assert the integrity of the feature compilation.
            assert len(rawFeatureTimes) == len(rawFeatures), \
                f"Found {len(rawFeatureTimes)} times and {len(rawFeatures)} features. These must be the same length."

            # Perform the feature averaging
            compiledFeatureTimes, compiledFeatures = self.averageFeatures_static(rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=0)
            compiledFeatureTimesHolders.append(compiledFeatureTimes)
            compiledFeatureHolders.append(compiledFeatures)
        # compiledFeatures dim: numBiomarkers, numTimePoints, numBiomarkerFeatures

        return compiledFeatureTimesHolders, compiledFeatureHolders
