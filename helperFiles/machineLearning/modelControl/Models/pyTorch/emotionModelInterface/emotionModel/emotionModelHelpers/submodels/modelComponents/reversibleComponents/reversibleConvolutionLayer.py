import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)  # The activation function to use for the neural operator.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.bounds = 1 / kernelSize  # The bounds for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.gradientScale = 1  # The scaling factor for the gradients.

        # The stability term to add to the diagonal.
        self.stabilityTerm = torch.eye(self.sequenceLength, dtype=torch.float64)*0.9

        # The restricted window for the neural weights.
        self.restrictedWindowMask = torch.ones(1, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        self.restrictedWindowMask = torch.tril(torch.triu(self.restrictedWindowMask, diagonal=-kernelSize//2 + 1), diagonal=kernelSize//2)
        self.restrictedWindowMask = self.restrictedWindowMask.repeat(repeats=(numSignals, 1, 1))  # Dim: 1, sequenceLength, sequenceLength

        # Calculate the offsets to map positions to kernel indices
        self.signalInds, self.rowInds, self.colInds = self.restrictedWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.colInds - self.rowInds + self.kernelSize // 2  # Adjust for kernel center

        # Initialize the neural weights.
        self.linearOperators = nn.ParameterList()
        for layerInd in range(self.numLayers):
            if self.kernelSize != self.sequenceLength: parameters = nn.Parameter(torch.randn(numSignals, kernelSize, dtype=torch.float64))
            else: parameters = nn.Parameter(torch.randn(numSignals, sequenceLength, sequenceLength, dtype=torch.float64))
            self.linearOperators.append(nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds))

        # Register hooks for each parameter in the list
        for param in self.linearOperators:
            param.register_hook(self.scaleGradients)

    def scaleGradients(self, grad):
        return grad * self.gradientScale

    def forward(self, inputData):
        # Cast the stability term to the device.
        self.restrictedWindowMask = self.restrictedWindowMask.to(inputData.device)
        self.stabilityTerm = self.stabilityTerm.to(inputData.device)

        for layerInd in range(self.numLayers):
            if self.forwardDirection:
                pseudoLayerInd = self.numLayers - layerInd - 1
                inputData = self.applyLayer(inputData, pseudoLayerInd)
                inputData = self.activationFunction(inputData, pseudoLayerInd % 2 == 0)
            else:
                inputData = self.activationFunction(inputData, layerInd % 2 == 0)
                inputData = self.applyLayer(inputData, layerInd)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Get the current neural weights information.
        neuralWeights = self.restrictedWindowMask.clone()
        kernelWeights = self.linearOperators[layerInd]
        # neuralWeight: numSignals, sequenceLength, sequenceLength
        # kernelWeights: numSignals, (kernelSize) or (sequenceLength, sequenceLength)

        if self.kernelSize != self.sequenceLength:
            # Gather the corresponding kernel values for each position
            kernel_values = kernelWeights[self.signalInds, self.kernelInds]  # Shape: (numIndices,)
            neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernel_values

        # Add a stability term to the diagonal.
        if self.kernelSize != self.sequenceLength: neuralWeights = neuralWeights + self.stabilityTerm
        else: neuralWeights = kernelWeights * (1 - self.stabilityTerm) + self.stabilityTerm

        # Backward direction: invert the neural weights.
        if self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Apply the neural weights to the input data.
        outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)

        return outputData
# 6, 13 -> 1.01
# 3, 7 -> 1.025
# 1, 3 -> 1.075


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 1024
    _activationMethod = 'nonLinearAddition'
    _kernelSize = 65
    _numLayers = 1

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
