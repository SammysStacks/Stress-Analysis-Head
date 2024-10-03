# General
import torch
import random

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


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
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.shape

        # Shuffle each tensor in the batch along the numSignals dimension
        if shuffle_indices is None: shuffle_indices = torch.randperm(numSignals)  # Generate random permutation indices for shuffling
        augmentedSignalData = signalData[:, shuffle_indices, :, :]  # Shuffle the signals

        return augmentedSignalData, shuffle_indices

    @staticmethod
    def getNewStartTimeIndices(signalData, minTimeWindow, maxTimeWindow):
        # signalData dim: [batchSize, numSignals, maxSequenceLength, numChannels]
        timeChannels = emotionDataInterface.getChannelData(signalData, modelConstants.timeChannel)  # Dim: (batchSize, numSignals, maxSequenceLength)
        batchSize, numSignals, maxSequenceLength = timeChannels.shape

        # Find the time window for the signal.
        newTimeWindow = generalMethods.biasedSample(minTimeWindow, maxTimeWindow, biasType="high") / modelConstants.maxTimeWindow
        targetTimes = torch.full((batchSize, numSignals, 1), newTimeWindow, device=timeChannels.device)

        # Find the start time indices for the signals.
        startTimeIndices = torch.searchsorted(-timeChannels, -targetTimes, side="left").squeeze(-1)  # Shape: (batchSize, numSignals)
        startTimeIndices = torch.clamp(startTimeIndices, min=0, max=maxSequenceLength-1)
        # startTimeIndices dim: [batchSize, numSignals]

        return startTimeIndices

    def changeNumSignals(self, signalData, minNumSignals=1, maxNumSignals=128, alteredDim=1):
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        numSignals = signalData.size(alteredDim)

        # Find a random place to cut the data.
        minNumSignals = max(minNumSignals + 1, int(numSignals / 3))
        randomEnd = int(generalMethods.biasedSample(minNumSignals, maxNumSignals, biasType="high"))

        # Expand the number of signals.
        repeat_times = (maxNumSignals + numSignals - 1) // numSignals  # Calculate the number of times we need to repeat the tensor
        signalData = signalData.repeat_interleave(repeat_times, dim=alteredDim)
        signalData = self.getInitialSignals(signalData, maxNumSignals)

        # Shuffle the signals to ensure that we are not always removing the same signals.
        signalData, shuffle_indices = self.shuffleDimension(signalData, shuffle_indices=None)
        signalData = self.getInitialSignals(signalData, randomEnd)

        return signalData
    
    @staticmethod
    def signalDropout(augmentedBatchData, dropoutPercent):
        # Assuming signalDatas is your tensor with dimensions [batchSize, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, sequenceLength, numChannels = augmentedBatchData.size()
        if dropoutPercent == 0: return augmentedBatchData

        # Find a random percentage to drop the data.
        finalDropoutPercent = generalMethods.biasedSample(range_start=0, range_end=dropoutPercent, biasType="low")
        dropoutMask = finalDropoutPercent < torch.rand((batchSize, numSignals, sequenceLength), device=augmentedBatchData.device)

        # Slice all the data at the same index
        augmentedBatchData = augmentedBatchData * dropoutMask.unsqueeze(-1)

        return augmentedBatchData

    @staticmethod
    def getRecentSignalPoints(signalData, finalLength):
        assert False  # return signalChannel[:, :, :finalLength].contiguous()

    @staticmethod
    def getInitialSignals(signalData, finalLength):
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        return signalData[:, 0:finalLength, :, :].contiguous()
