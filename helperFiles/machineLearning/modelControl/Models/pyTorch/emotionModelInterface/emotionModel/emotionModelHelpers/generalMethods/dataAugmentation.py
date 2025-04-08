# General
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class dataAugmentation:

    @staticmethod
    def addNoise(data, noiseSTD=0.01):
        # If we are training, add noise to the final state to ensure continuity of the latent space.
        return data + torch.randn_like(data, device=data.device) * noiseSTD

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
        dropoutMask = dropoutMask.expand_as(signalData)

        # Apply the mask to the data
        augmentedData = signalData * dropoutMask

        return augmentedData

    @staticmethod
    def signalDropout(signalData, dropoutPercent):
        # Assuming signalData is your tensor with dimensions [batchSize, numSignals, sequenceLength, numChannels]
        batchSize, numSignals, sequenceLength, numChannels = signalData.size()
        if dropoutPercent == 0: return signalData

        # Create the dropout mask with the required shape directly
        dropoutMask = dropoutPercent < torch.rand((batchSize, numSignals, sequenceLength, 1), device=signalData.device)

        # Apply the dropout mask without needing unsqueeze
        augmentedData = signalData * dropoutMask

        return augmentedData
