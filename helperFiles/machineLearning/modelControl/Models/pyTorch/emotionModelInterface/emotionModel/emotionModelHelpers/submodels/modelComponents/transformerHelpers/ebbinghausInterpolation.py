# Pytorch

import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights


class ebbinghausInterpolation(emotionModelWeights):

    def __init__(self, numSignals, encodedDimension):
        super(ebbinghausInterpolation, self).__init__()
        # General model parameters.
        self.encodedTimeWindow = modelConstants.timeWindows[-1]  # The time window for the encoded signal.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numSignals = numSignals  # The number of signals to interpolate.

        # Define the parameters.
        self.ebbinghausWeights = self.timeDependantSignalWeights(numSignals)
        pseudoEncodedTimes = torch.arange(0, self.encodedTimeWindow, step=self.encodedTimeWindow/self.encodedDimension, device=self.ebbinghausWeights.device)
        self.register_buffer(name='pseudoEncodedTimes', tensor=torch.flip(pseudoEncodedTimes, dims=[0]))

        # Extra components.
        self.selfAttention = self.linearModel(numInputFeatures=encodedDimension, numOutputFeatures=encodedDimension, activationMethod='none', layerType='fc', addBias=False)

    def forward(self, signalData):
        # signalData dimension: [batchSize, numSignals, maxSequenceLength, numChannels]
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        missingDataMask = torch.as_tensor((datapoints == 0) & (timepoints == 0), device=datapoints.device)
        # datapoints and timepoints: [batchSize, numSignals, maxSequenceLength)
        # timepoints: [further away from survey (300) -> closest to survey (0)]

        # Calculate the decay time weights.
        deltaTimes = self.pseudoEncodedTimes - timepoints.unsqueeze(-1)
        decayWeights = self.ebbinghausDecayExp(deltaTimes, self.ebbinghausWeights)
        # deltaTimes and decayWeights: [batchSize, numSignals, maxSequenceLength, encodedDimension]

        # Add self-attention to the decay weights.
        # decayWeights = self.selfAttention(decayWeights)

        # Normalize the weights.
        decayWeights = self.sparseSoftmax(decayWeights.clone(), missingDataMask.unsqueeze(-1), dim=-1)
        # decayWeights: [batchSize, numSignals, maxSequenceLength, encodedDimension]

        # Combine the attention values.
        finalSignalData = torch.matmul(datapoints.unsqueeze(2), decayWeights).squeeze(2)
        # finalSignalData: [batchSize, numSignals, encodedDimension]

        # Standardize the data.
        finalSignalData = self.standardizeData(datapoints, finalSignalData)
        # finalSignalData: [batchSize, numSignals, encodedDimension]

        return finalSignalData, missingDataMask

    @staticmethod
    def standardizeData(datapoints, finalSignalData):
        # Calculate the original mean and variance of the datapoints (for maintaining).
        original_variance = datapoints.var(dim=-1, keepdim=True, unbiased=False)  # Variance over maxSequenceLength
        original_mean = datapoints.mean(dim=-1, keepdim=True)  # Mean over maxSequenceLength

        # Calculate the mean and variance of the finalSignalData.
        finalSignalData_variance = finalSignalData.var(dim=-1, keepdim=True, unbiased=False)  # Variance over encodedDimension
        finalSignalData_mean = finalSignalData.mean(dim=-1, keepdim=True)  # Mean over encodedDimension

        # Standardize the finalSignalData.
        standardizedSignalData = (finalSignalData - finalSignalData_mean) / (1e-10 + finalSignalData_variance.sqrt())
        standardizedSignalData = standardizedSignalData * original_variance.sqrt() + original_mean

        return standardizedSignalData

    @staticmethod
    def sparseNorm(data, mask, dim):
        # Calculate the normalization factor.
        normalizationFactor = data.sum(dim=dim, keepdim=True)
        fullMask = mask.expand(-1, -1, -1, data.size(-1))

        # Perform softmax with absent data as exp(-inf) = 0.
        normData = data / (1e-10 + normalizationFactor)
        normData = normData.masked_fill(fullMask, 0)

        return normData

    @staticmethod
    def sparseSoftmax(data, mask, dim):
        # Calculate the exponential of the data.
        softmaxData = (data - data.max(dim=dim, keepdim=True)[0]).exp()
        fullMask = mask.expand(-1, -1, -1, softmaxData.size(-1))
        softmaxData = softmaxData.masked_fill(fullMask, 0)

        # Perform softmax with absent data as exp(-inf) = 0.
        softmaxData = softmaxData / (1e-10 + softmaxData.sum(dim=dim, keepdim=True))
        softmaxData = softmaxData.masked_fill(fullMask, 0)

        return softmaxData


if __name__ == '__main__':
    # General parameters.
    _encodedDimension = 512
    _signalDimension = 128
    _numSignals = 64
    _batchSize = 4

    # Set up the parameters.
    _signalDataTimes = torch.arange(0, _signalDimension, step=1)
    _signalData = torch.randn((_batchSize, _numSignals, _signalDimension, 2))
    _signalData[:, :, :, 0] = _signalDataTimes
    _signalData[:, :, 300:, :] = 0

    # Initialize the model.
    ebbinghausModel = ebbinghausInterpolation(numSignals=_numSignals, encodedDimension=_encodedDimension)
    _finalSignalData = ebbinghausModel(_signalData)
