# General
import torch
import random

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.classWeightHelpers import classWeightHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class emotionDataInterface:

    @staticmethod
    def addNoise(data, trainingFlag, noiseSTD=0.01):
        # If we are training, add noise to the final state to ensure continuity of the latent space.
        return data + torch.randn_like(data, device=data.device) * noiseSTD if trainingFlag or noiseSTD == 0 else data

    @staticmethod
    def getTimeIntervalInd(timeData, timePoint, mustIncludeTimePoint=False):
        # timeData is a torch array of size (maxSequenceLength)
        # Assert the validity of the input parameters.
        assert 0 <= timePoint, f"Expected a positive time point, but got {timePoint}"
        timeData = torch.as_tensor(timeData)  # Ensure timeData is a torch tensor

        # Find the index of the time point in the time data
        timeInd = torch.where(timePoint <= timeData)[0]
        if len(timeInd) == 0: return 0

        # Determine if the time point is included in the time data
        isTimePointIncluded = timeData[0] <= timePoint
        timeInd = timeInd[0].item()

        # Include the time point if necessary
        if not isTimePointIncluded and mustIncludeTimePoint:
            timeInd = max(timeInd - 1, 0)

        return timeInd

    @staticmethod
    def shuffleDimension(signalDatas):
        # Get the shape of the tensors
        batchSize, numSignals, maxSequenceLength, numChannels = signalDatas[0].shape

        # Generate random permutation indices for shuffling
        shuffle_indices = torch.randperm(numSignals)

        # Shuffle each tensor in the batch along the numSignals dimension
        augmentedSignalDatas = (signalData[:, shuffle_indices, :, :] for signalData in signalDatas)

        return augmentedSignalDatas

    def changeSignalLength(self, minimumSignalLength, signalDatas):
        # Assuming signalDatas is your tensor with dimensions [numExperiments, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, sequenceLength, numChannels = signalDatas[0].shape

        # Find a random place to cut the data.
        randomSignalEnd = torch.tensor(generalMethods.biased_high_sample(minimumSignalLength, sequenceLength, randomValue=random.uniform(0, 1)), dtype=torch.int32).item()

        # Slice all the data at the same index
        augmentedSignalDatas = (self.getRecentSignalPoints(signalData, randomSignalEnd) for signalData in signalDatas)

        return augmentedSignalDatas

    @staticmethod
    def getRandomTimeInterval(minTimeWindow, maxTimeWindow):
        return torch.tensor(generalMethods.biased_high_sample(minTimeWindow, maxTimeWindow, randomValue=random.uniform(a=0, b=1)), dtype=torch.int32).item()

    def getRandomSignalCutOff(self, allSignalTimes, minTimeWindow, maxTimeWindow):
        # allSignalTimes: A torch array of size (batchSize, numSignals, maxSequenceLength)
        newTimeWindow = self.getRandomTimeInterval(minTimeWindow, maxTimeWindow)
        batchSize, numSignals, maxSequenceLength = allSignalTimes.size()

        # Find the random starting point for the signal.
        allStartSignalInds = torch.zeros(batchSize, numSignals, dtype=torch.int32)

        # For each batch.
        for batchInd in range(batchSize):
            eachSignalTimes = allSignalTimes[batchInd]  # Dim: (numSignals, maxSequenceLength)

            # For each signal in the batch.
            for signalInd in range(numSignals):
                # Find the number of points in the signal.
                signalTimes = eachSignalTimes[signalInd]  # Dim: (maxSequenceLength)

                # Find the time window for the signal.
                allStartSignalInds[batchInd, signalInd] = self.getTimeIntervalInd(signalTimes, timePoint=newTimeWindow, mustIncludeTimePoint=False)

        return allStartSignalInds

    def changeNumSignals(self, signalDatas, minNumSignals, maxNumSignals, alteredDim=1):
        # Assuming signalDatas is your tensor with dimensions [numCopies, numExperiments, numSignals, maxSequenceLength, numChannels]
        numSignals = signalDatas[0].size(alteredDim)
        minValue = max(minNumSignals + 1, int(numSignals / 3))
        repeat_times = (maxNumSignals + numSignals - 1) // numSignals  # Calculate the number of times we need to repeat the tensor

        # Expand the number of signals.
        signalDatas = torch.stack(signalDatas).repeat_interleave(repeat_times, dim=alteredDim)[:, :maxNumSignals, :, :]

        # Shuffle the signals to ensure that we are not always removing the same signals.
        signalDatas = self.shuffleDimension(signalDatas)

        # Find a random place to cut the data.
        randomEnd = torch.tensor(generalMethods.biased_high_sample(minValue, maxNumSignals, randomValue=random.uniform(a=0, b=1)), dtype=torch.int32).item()

        finalDatas = []
        for signalData in signalDatas:
            # Slice all the data at the same index
            finalDatas.append(self.getInitialSignals(signalData, randomEnd))

        return finalDatas

    @staticmethod
    def getRecentSignalPoints(signalData, finalLength):
        assert False  # return signalData[:, :, :finalLength].contiguous()

    @staticmethod
    def getInitialSignals(signalData, finalLength):
        return signalData[:, 0:finalLength, :, :].contiguous()

    # ---------------------------------------------------------------------- #
    # ---------------------- Data Structure Interface ---------------------- #

    @staticmethod
    def indexSubjectIdentifiers(subjectIdentifiers, subjectIdentifierName):
        # subjectIdentifiers: A torch array of size (batchSize, numSubjectIdentifiers)
        subjectIdentifierInd = modelConstants.subjectIdentifiers.index(subjectIdentifierName)

        return subjectIdentifiers[:, subjectIdentifierInd]

    @staticmethod
    def separateData(inputData):
        # allSignalData: A torch array of size (batchSize, numSignals, maxSequenceLength + numSubjectIdentifiers, [signalData, previousSignalPoints, nextDeltaTimes, previousDeltaTimes, nextDeltaTimes, time])
        # Extract the incoming data's dimension and ensure a proper data format.
        batchSize, numSignals, signalInfoLength, numChannels = inputData.size()

        # Find the number of subject identifiers.
        numSubjectIdentifiers = len(modelConstants.subjectIdentifiers)
        maxSequencePoints = signalInfoLength - numSubjectIdentifiers

        # Assert the validity of the input data
        assert signalInfoLength == maxSequencePoints + numSubjectIdentifiers, \
            f"{signalInfoLength} != {maxSequencePoints} {numSubjectIdentifiers}"

        # Separate the sequence and demographic information.
        signalData = inputData[:, :, 0:maxSequencePoints, 0:len(modelConstants.signalChannelNames)]
        signalTimes = inputData[:, :, 0:maxSequencePoints, len(modelConstants.signalChannelNames)]
        subjectIdentifiers = inputData[:, 0, maxSequencePoints:signalInfoLength, 0]
        # signalData dimension: batchSize, numSignals, maxSequenceLength, [signalData, previousSignalPoints, nextDeltaTimes, previousDeltaTimes, nextDeltaTimes]
        # signalTimes dimension: batchSize, numSignals, maxSequenceLength
        # subjectInds dimension: batchSize, numSubjectIdentifiers

        return signalTimes, signalData, subjectIdentifiers

    def getReconstructionIndex(self, allTrainingMasks):
        # Find the first label index with training points.
        reconstructionIndices = self.getLabelInds_withPoints(allTrainingMasks)
        assert len(reconstructionIndices) != 0, f"We should have some training data: {reconstructionIndices}"

        return reconstructionIndices[0]

    @staticmethod
    def getActivityColumn(allLabels, activityLabelInd):
        return allLabels[:, activityLabelInd]

    @staticmethod
    def getEmotionMasks(allLabelsMask, numEmotions):
        return allLabelsMask[:, 0:numEmotions]

    @staticmethod
    def getEmotionColumn(allLabels, emotionInd):
        return allLabels[:, emotionInd]

    def getActivityLabels(self, allLabels, allLabelsMask, activityLabelInd):
        activityMask = self.getActivityColumn(allLabelsMask, activityLabelInd)
        allActivityLabels = self.getActivityColumn(allLabels, activityLabelInd)

        return allActivityLabels[activityMask]

    def getEmotionLabels(self, emotionInd, allLabels, allLabelsMask):
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)
        emotionLabels = self.getEmotionColumn(allLabels, emotionInd)

        return emotionLabels[emotionDataMask]

    def getEmotionDistributions(self, emotionInd, predictedEmotionLabels, allLabels, allLabelsMask, allEmotionClasses, emotionLength):
        # Organize the emotion's training information.
        emotionLabels = self.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)

        # Get the predicted and true emotion distributions.
        predictedTrainingEmotions = predictedEmotionLabels[emotionInd][emotionDataMask]
        trueTrainingEmotions = classWeightHelpers.gausEncoding(emotionLabels, allEmotionClasses[emotionInd], emotionLength)

        return predictedTrainingEmotions, trueTrainingEmotions

    @staticmethod
    def getLabelInds_withPoints(allTrainingMasks):
        # Find the label indices with training points.
        numTrainingPoints = allTrainingMasks.sum(dim=0)  # Find the number of training points per label.
        goodExperimentalInds = numTrainingPoints.nonzero()
        # If there are no good indices, return an empty list.
        if len(goodExperimentalInds) == 0: return []

        return goodExperimentalInds[:, 0]
