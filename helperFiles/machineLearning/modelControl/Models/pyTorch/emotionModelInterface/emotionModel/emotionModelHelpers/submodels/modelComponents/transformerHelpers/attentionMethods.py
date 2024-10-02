# Pytorch
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderHelpers.signalEncoderModules import signalEncoderModules


class attentionMethods(signalEncoderModules):

    def __init__(self,  inputQueryKeyDim, latentQueryKeyDim, inputValueDim, latentValueDim, numHeads, addBias=False):
        super(attentionMethods, self).__init__()
        # General model parameters.
        self.latentQueryKeyDim = latentQueryKeyDim  # The embedded dimension of the query and keys: Int
        self.inputQueryKeyDim = inputQueryKeyDim  # The initial dimension of the query and keys: Int
        self.latentValueDim = latentValueDim  # The embedded dimension of the values: Int
        self.inputValueDim = inputValueDim  # The initial dimension of the values: Int
        self.numHeads = numHeads  # The number fo splits to any token: Int
        self.addBias = addBias    # Adds a bias to any weights. Default: True.

        # Asser the integrity of the inputs.
        assert latentQueryKeyDim % numHeads == 0, "The query and key dimension must be divisible by the number of heads."
        assert inputQueryKeyDim == 1, "Not yet implemented for multiple query and key dimensions."
        assert numHeads == 1, "Not yet implemented for multiple heads."

        # Define the attention parameters.
        self.queryWeights = self.linearModel(numInputFeatures=self.inputQueryKeyDim, numOutputFeatures=self.latentQueryKeyDim, activationMethod='none', layerType='fc', addBias=self.addBias)
        self.keyWeights = self.linearModel(numInputFeatures=self.inputQueryKeyDim, numOutputFeatures=self.latentQueryKeyDim, activationMethod='none', layerType='fc', addBias=self.addBias)
        self.valueWeights = self.linearModel(numInputFeatures=self.inputValueDim, numOutputFeatures=self.latentValueDim, activationMethod='none', layerType='fc', addBias=self.addBias)

        # Define the attention parameters.
        self.queryTimeInfluence = self.linearModel(numInputFeatures=self.inputQueryKeyDim, numOutputFeatures=self.latentValueDim, activationMethod='none', layerType='fc', addBias=self.addBias)

    def forward(self, signalData):
        # signalData dimension: [batchSize, numSignals, maxSequenceLength, numChannels]
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel).unsqueeze(-1)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel).unsqueeze(-1)
        validDataMask = torch.as_tensor(((datapoints == 0) & (timepoints == 0)))
        # datapoints and timepoints dimension: [batchSize, numSignals, maxSequenceLength, inputQueryKeyDim = 1]

        # Calculate the query, key, and value.
        query = self.queryWeights(timepoints) + timepoints  # Dimension: [batchSize, numSignals, maxSequenceLength, latentQueryKeyDim]
        value = self.valueWeights(datapoints) + datapoints  # Dimension: [batchSize, numSignals, maxSequenceLength, latentValueDim]
        key = self.keyWeights(timepoints) + timepoints  # Dimension: [batchSize, numSignals, maxSequenceLength, latentQueryKeyDim]

        # Calculate the weight of each token.
        similarityScores = torch.matmul(query, key.transpose(-2, -1)) / (self.latentQueryKeyDim ** 0.5)
        # similarityScores dimension: [batchSize, numSignals, maxSequenceLength, maxSequenceLength]

        # Calculate the attention weights.
        attentionWeights = self.sparseSoftmax(similarityScores, validDataMask, dim=-1)
        # attentionWeights meaning: given a token in maxSequenceLength, how similar is it to all other tokens.
        # attentionWeights: [batchSize, numSignals, maxSequenceLength, maxSequenceLength]

        # Add in each datapoints context.
        selfAttentionValues = torch.matmul(attentionWeights, value) + datapoints
        # selfAttentionValues dimension: [batchSize, numSignals, maxSequenceLength, latentValueDim=finalOutputDim]
        # selfAttentionValues meaning: given a token in maxSequenceLength, what is its value for the final output.

        # See how the token is related to the other tokens.
        timeInfluence = self.queryTimeInfluence(timepoints) + timepoints  # Dimension: [batchSize, numSignals, maxSequenceLength, latentValueDim=finalOutputDim]
        self.sparseSoftmax(timeInfluence, validDataMask, dim=2)

        # Combine the attention values.
        finalSignalData = (selfAttentionValues * timeInfluence).sum(dim=2)
        # finalSignalData: [batchSize, numSignals, latentValueDim=finalOutputDim]

        return finalSignalData

    @staticmethod
    def sparseSoftmax(data, mask, dim):
        # Perform softmax with absent data as exp(-inf) = 0.
        data = data.masked_fill(mask=mask, value=float('-inf'))
        data = torch.softmax(data, dim=dim)

        # Mask out the padding tokens.
        data = data.masked_fill(mask=mask, value=0)

        return data
