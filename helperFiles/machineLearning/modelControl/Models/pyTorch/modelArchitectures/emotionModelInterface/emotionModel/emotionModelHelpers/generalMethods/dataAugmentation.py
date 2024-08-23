# General
import torch
import random

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.classWeightHelpers import classWeightHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class dataAugmentation:

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
    def shuffleDimension(signalData, shuffle_indices=None):
        # Get the shape of the tensors
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.shape

        # Shuffle each tensor in the batch along the numSignals dimension
        if shuffle_indices is not None: shuffle_indices = torch.randperm(numSignals)  # Generate random permutation indices for shuffling
        augmentedSignalData = signalData[:, shuffle_indices, :, :]  # Shuffle the signals

        return augmentedSignalData, shuffle_indices

    def changeSignalLength(self, minimumSignalLength, signalDatas):
        # Assuming signalDatas is your tensor with dimensions [batchSize, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, sequenceLength, numChannels = signalDatas[0].shape

        # Find a random place to cut the data.
        randomSignalEnd = torch.tensor(generalMethods.biased_high_sample(minimumSignalLength, sequenceLength, randomValue=random.uniform(0, 1)), dtype=torch.int32).item()

        # Slice all the data at the same index
        augmentedSignalDatas = (self.getRecentSignalPoints(signalData, randomSignalEnd) for signalData in signalDatas)

        return augmentedSignalDatas

    @staticmethod
    def getRandomTimeInterval(minTimeWindow, maxTimeWindow):
        return torch.tensor(generalMethods.biased_high_sample(minTimeWindow, maxTimeWindow, randomValue=random.uniform(a=0, b=1)), dtype=torch.int32).item()

    def getNewStartTimeIndices(self, signalData, minTimeWindow, maxTimeWindow):
        # signalData dim: [batchSize, numSignals, maxSequenceLength, numChannels]
        # metaInfo dim: [batchSize, numMetadata]
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()

        # Find the number of signal points and time window.
        timeChannels = emotionDataInterface.getChannelData(signalData, modelConstants.timeChannel)  # Dim: (batchSize, numSignals, maxSequenceLength)

        # Find the time window for the signal.
        newTimeWindow = self.getRandomTimeInterval(minTimeWindow, maxTimeWindow)
        numWindowPoints = (timeChannels <= newTimeWindow).sum(dim=-1)
        startTimeIndices = maxSequenceLength - numWindowPoints

        return startTimeIndices

    def changeNumSignals(self, *signalDatas, minNumSignals=1, maxNumSignals=128, alteredDim=1):
        # Assuming signalDatas is your tensor with dimensions [numCopies, batchSize, numSignals, maxSequenceLength, numChannels]
        numSignals = signalDatas[0].size(alteredDim)

        # Find a random place to cut the data.
        minNumSignals = max(minNumSignals + 1, int(numSignals / 3))
        randomEnd = int(generalMethods.biased_high_sample(minNumSignals, maxNumSignals, randomValue=random.uniform(a=0, b=1)))

        # Expand the number of signals.
        repeat_times = (maxNumSignals + numSignals - 1) // numSignals  # Calculate the number of times we need to repeat the tensor

        # Set up the signal augmentation.
        shuffle_indices = None
        finalDatas = []

        # For each signal data.
        for signalData in signalDatas:
            # Expand the number of signals.
            signalData = signalData.repeat_interleave(repeat_times, dim=alteredDim)[:, :maxNumSignals, :, :]

            # Shuffle the signals to ensure that we are not always removing the same signals.
            signalData, shuffle_indices = self.shuffleDimension(signalData, shuffle_indices)

            # Slice all the data at the same index
            finalDatas.append(self.getInitialSignals(signalData, randomEnd))

        return finalDatas

    @staticmethod
    def getRecentSignalPoints(signalData, finalLength):
        assert False  # return signalChannel[:, :, :finalLength].contiguous()

    @staticmethod
    def getInitialSignals(signalData, finalLength):
        return signalData[:, 0:finalLength, :, :].contiguous()
