import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions


class reversibleConvolution(reversibleInterface):

    def __init__(self, numChannels, kernelSize, activationMethod, numLayers, skipConnection):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)  # The activation function to use for the neural operator.
        self.skipConnection = skipConnection  # Whether to use a skip connection. THIS IS NOT A TRADITIONAL SKIP CONNECTION!!
        self.numChannels = numChannels  # The number of channels in the input signal.
        self.kernelSize = kernelSize  # The size of the kernel for the convolution.
        self.numLayers = numLayers  # The number of layers in the reversible convolution.

        # Initialize the neural weights.
        self.linearOperators = nn.ModuleList()
        for layerInd in range(self.numLayers):
            self.linearOperators.append(nn.Conv1d(in_channels=numChannels, out_channels=numChannels, kernel_size=kernelSize, stride=1, padding=0, dilation=1, groups=numChannels, padding_mode='reflect', bias=False))

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2  # Assuming dilation = 1
        assert self.kernelSize % 2 == 1, "The kernel size must be odd."

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if self.forwardDirection:
                inputData = self.applyLayer(inputData, self.numLayers - layerInd - 1)
                inputData = self.activationFunction(inputData)
            else:
                inputData = self.activationFunction(inputData)
                inputData = self.applyLayer(inputData, layerInd)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Prepare the neural weights.
        neuralWeights = self.linearOperators[layerInd].weight.data.clone()
        kernelSize = neuralWeights.size(-1)

        # Create a square neural weight matrix.
        neuralWeights = neuralWeights.unsqueeze(-1).repeat(1, 1, 1, kernelSize).transpose(2, 3)
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Create an invertible neural weight matrix.
        stabilityTerm = self.getStabilityTerm(kernelSize, scalingFactor=1, device=inputData.device)
        neuralWeights = torch.triu(neuralWeights, diagonal=0) + stabilityTerm
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Skip connection.
        skipConnectionWeight = torch.zeros(kernelSize, device=inputData.device)
        skipConnectionWeight[kernelSize//2] = 1/self.numLayers if self.skipConnection else 0
        neuralWeights = neuralWeights + skipConnectionWeight

        # Invert the neural weights if needed.
        if not self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Perform the convolution.
        convolutionalData = torch.nn.functional.conv1d(inputData, neuralWeights[:, :, :, 0], bias=None, stride=1, padding=kernelSize, dilation=1, groups=self.numChannels)
        convolutionalData = convolutionalData[:, :, kernelSize:-1]
        convolutionalData.half()

        return convolutionalData

    def combineNeuralWeights(self):
        # Extract the first set of weights.
        neuralWeights = self.linearOperators[0].weight.data.clone()

        # For each layer.
        for layerInd in range(self.numLayers):
            if layerInd == 0:
                neuralWeights = neuralWeights.permute(1, 0, 2)
            else:
                # Extract the next set of weights and flip them.
                nextWeights = self.linearOperators[layerInd].weight.data.clone()
                nextWeights = torch.flip(nextWeights, dims=[-1])  # Flip the kernel dimension.

                # Perform the convolution.
                neuralWeights = torch.nn.functional.conv1d(neuralWeights, nextWeights, bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numChannels)
                neuralWeights = neuralWeights[:, :, 1:-1]
        neuralWeights = neuralWeights.permute(1, 0, 2)

        return neuralWeights


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 4, 128, 512
    _activationMethod = 'reversibleLinearSoftSign_2_0.9'
    _numLayers = 25
    _kernelSize = 5

    # Set up the parameters.
    neuralLayerClass = reversibleConvolution(numChannels=_numSignals, kernelSize=_kernelSize, activationMethod=_activationMethod, numLayers=_numLayers, skipConnection=True)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float32)
    _inputData = _inputData - _inputData.min()
    _inputData = _inputData / _inputData.max()
    _inputData = 2*_inputData - 1

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-4)
