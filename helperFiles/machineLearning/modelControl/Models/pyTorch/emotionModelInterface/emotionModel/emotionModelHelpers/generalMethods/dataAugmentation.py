# General
import torch

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
        newTimeWindow = generalMethods.biasedSample(minTimeWindow, maxTimeWindow, biasType="high")
        targetTimes = torch.full((batchSize, numSignals, 1), newTimeWindow, device=timeChannels.device)

        # Find the start time indices for the signals.
        startTimeIndices = torch.searchsorted(-timeChannels, -targetTimes, side="left").squeeze(-1)  # Shape: (batchSize, numSignals)
        startTimeIndices = torch.clamp(startTimeIndices, min=0, max=maxSequenceLength-1)
        # startTimeIndices dim: [batchSize, numSignals]

        return startTimeIndices

    @staticmethod
    def changeNumSignals(signalData, dropoutPercent):
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        if dropoutPercent == 0: return signalData

        # Create a mask to drop p% of the signals
        dropoutMask = dropoutPercent < torch.rand(batchSize, numSignals, 1, 1, device=signalData.device)  # Randomly keep (1-p)% of the signals

        # Expand the mask to cover all timestamps and channels
        dropoutMask = dropoutMask.expand(batchSize, numSignals, maxSequenceLength, 2)

        # Apply the mask to the data
        augmentedData = signalData * dropoutMask

        return augmentedData
    
    @staticmethod
    def signalDropout(signalData, dropoutPercent):
        # Assuming signalDatas is your tensor with dimensions [batchSize, numSignals, maxSequenceLength, numChannels]
        batchSize, numSignals, sequenceLength, numChannels = signalData.size()
        if dropoutPercent == 0: return signalData

        # Find a random percentage to drop the data.
        dropoutMask = dropoutPercent < torch.rand((batchSize, numSignals, sequenceLength), device=signalData.device)

        # Slice all the data at the same index
        augmentedData = signalData * dropoutMask.unsqueeze(-1)

        return augmentedData

    @staticmethod
    def getInitialSignals(signalData, finalLength):
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        return signalData[:, 0:finalLength, :, :].contiguous()
