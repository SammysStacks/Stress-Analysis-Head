import math

import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface

# DEPRECATED: WILL NOT WORK!
class reversibleConvolutionFFT(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, activationMethod, numLayers):
        super(reversibleConvolutionFFT, self).__init__()
        # General parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)  # The activation function to use for the neural operator.
        self.sequenceLength = sequenceLength  # The size of the kernel for the convolution.
        self.numSignals = numSignals  # The number of channels in the input signal.
        self.numLayers = numLayers  # The number of layers in the reversible convolution.
        self.epsilon = 0.1  # The epsilon value to add to the neural weights.
        assert False

        # Initialize the neural weights.
        self.linearOperators = nn.ParameterList()
        for layerInd in range(self.numLayers):
            parameters = nn.Parameter(torch.randn(numSignals, int(sequenceLength/2 + 1), dtype=torch.float64))
            self.linearOperators.append(nn.init.uniform_(parameters, a=-math.sqrt(3), b=math.sqrt(3)))

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
        # Prepare the neural weights.
        neuralWeights = self.linearOperators[layerInd]

        # Project the data into the Fourier domain.
        fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='ortho')

        if self.forwardDirection: fourierData = fourierData * neuralWeights
        else: fourierData = fourierData / neuralWeights

        # Return to physical space
        outputData = torch.fft.irfft(fourierData, n=self.sequenceLength, dim=-1, norm='ortho')
        # outputData dimension: batchSize, numOutputChannels, signalDimension

        return outputData

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
                neuralWeights = torch.nn.functional.conv1d(neuralWeights, nextWeights, bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numSignals)
                neuralWeights = neuralWeights[:, :, 1:-1]
        neuralWeights = neuralWeights.permute(1, 0, 2)

        return neuralWeights


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 4, 128, 512
    _activationMethod = 'nonLinearMultiplication'
    _numLayers = 100
    _kernelSize = 3

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionFFT(numSignals=_numSignals, sequenceLength=_sequenceLength, activationMethod=_activationMethod, numLayers=_numLayers)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
    _inputData = _inputData - _inputData.min()
    _inputData = _inputData / _inputData.max()
    _inputData = 2*_inputData - 1

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
