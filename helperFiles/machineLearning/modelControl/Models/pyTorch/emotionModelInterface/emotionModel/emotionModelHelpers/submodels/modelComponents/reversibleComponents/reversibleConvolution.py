import random

import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions

# NOT WORKING YET!
class reversibleConvolution(reversibleInterface):

    def __init__(self, numChannels, kernelSize, activationMethod, numLayers):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)  # The activation function to use for the neural operator.
        self.numChannels = numChannels  # The number of channels in the input signal.
        self.kernelSize = kernelSize  # The size of the kernel for the convolution.
        self.numLayers = numLayers  # The number of layers in the reversible convolution.

        # Initialize the neural weights.
        self.linearOperators = nn.ModuleList()
        for layerInd in range(self.numLayers):
            self.linearOperators.append(nn.Conv1d(in_channels=numChannels, out_channels=numChannels, kernel_size=kernelSize, stride=1, padding=0, dilation=1, groups=numChannels, padding_mode='reflect', bias=False, dtype=torch.float64))

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2  # Assuming dilation = 1
        assert self.kernelSize % 2 == 1, "The kernel size must be odd."

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if self.forwardDirection:
                pseudoLayerInd = self.numLayers - layerInd - 1
                inputData = self.applyLayer(inputData, pseudoLayerInd)
                # inputData = self.activationFunction(inputData, pseudoLayerInd % 2 == 0)
            else:
                # inputData = self.activationFunction(inputData, layerInd % 2 == 0)
                inputData = self.applyLayer(inputData, layerInd)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Create a square neural weight matrix.
        neuralWeights = self.linearOperators[layerInd].weight.unsqueeze(-1)
        neuralWeights = torch.matmul(neuralWeights, neuralWeights.transpose(2, 3))
        # numInputChannels, numOutputChannels, kernelSize, kernelSize

        # Add a stability term to the diagonal.
        # TODO: The stability term does not yet ensure convertibility.
        stabilityTerm = self.getStabilityTerm(self.kernelSize, scalingFactor=1, device=inputData.device)
        neuralWeights = torch.triu(neuralWeights, diagonal=0) + stabilityTerm.unsqueeze(0).unsqueeze(0)
        # neuralWeights = neuralWeights / neuralWeights.norm(dim=-1, keepdim=True)

        # Skip connection.
        # neuralWeights = torch.linalg.qr(neuralWeights, mode='reduced')[1]
        # neuralWeights[:, :, 0, self.kernelSize//2] = 1  # Skip connection.
        print(neuralWeights[0, 0], neuralWeights[0, 0, self.kernelSize//2, :])

        # Invert the neural weights if needed.
        if not self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Perform the convolution.
        convolutionalData = torch.nn.functional.conv1d(inputData, neuralWeights[:, :, :, -1], bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numChannels)
        convolutionalData = convolutionalData[:, :, self.kernelSize:-1]

        return convolutionalData

    def combineNeuralWeights(self):
        # Extract the first set of weights.
        neuralWeights = self.linearOperators[0].weight

        # For each layer.
        for layerInd in range(self.numLayers):
            if layerInd == 0:
                neuralWeights = neuralWeights.permute(1, 0, 2)
            else:
                # Extract the next set of weights and flip them.
                nextWeights = self.linearOperators[layerInd].weight
                nextWeights = torch.flip(nextWeights, dims=[-1])  # Flip the kernel dimension.

                # Perform the convolution.
                neuralWeights = torch.nn.functional.conv1d(neuralWeights, nextWeights, bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numChannels)
                neuralWeights = neuralWeights[:, :, 1:-1]
        neuralWeights = neuralWeights.permute(1, 0, 2)

        return neuralWeights


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 4, 128, 512
    _activationMethod = 'nonLinearAddition'
    _numLayers = 1
    _kernelSize = 3

    random.seed(42)

    # Set up the parameters.
    neuralLayerClass = reversibleConvolution(numChannels=_numSignals, kernelSize=_kernelSize, activationMethod=_activationMethod, numLayers=_numLayers)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
    _inputData = _inputData - _inputData.min()
    _inputData = _inputData / _inputData.max()
    _inputData = 2*_inputData - 1

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData.round(decimals=6), atol=1e-6, numLayers=1)
