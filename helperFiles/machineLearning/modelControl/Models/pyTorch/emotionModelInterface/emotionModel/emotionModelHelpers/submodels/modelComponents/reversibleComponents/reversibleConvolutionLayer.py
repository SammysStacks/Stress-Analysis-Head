import math

import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod, switchActivationDirection):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.bounds = 1 / kernelSize  # The bounds for the neural weights.

        # The stability term to add to the diagonal.
        self.stabilityTerm = torch.eye(self.sequenceLength, dtype=torch.float64)

        # The restricted window for the neural weights.
        self.restrictedWindowMask = torch.ones(1, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        self.restrictedWindowMask = torch.tril(torch.triu(self.restrictedWindowMask, diagonal=-kernelSize//2 + 1), diagonal=kernelSize//2)
        self.restrictedWindowMask = self.restrictedWindowMask.repeat(repeats=(numSignals, 1, 1))  # Dim: 1, sequenceLength, sequenceLength
        assert kernelSize <= sequenceLength, f"The kernel size is larger than the sequence length: {kernelSize}, {sequenceLength}"

        # Calculate the offsets to map positions to kernel indices
        self.signalInds, self.rowInds, self.colInds = self.restrictedWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.colInds - self.rowInds + self.kernelSize // 2  # Adjust for kernel center

        # Initialize the neural layers.
        self.activationFunctions,  self.linearOperators = nn.ModuleList(), nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            parameters = nn.Parameter(torch.randn(numSignals, kernelSize, dtype=torch.float64))
            self.linearOperators.append(nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds))

            # Add the activation function.
            activationMethod = f"{self.activationMethod}_{switchActivationDirection}"
            self.activationFunctions.append(activationFunctions.getActivationMethod(activationMethod))
            switchActivationDirection = not switchActivationDirection

        # Register hooks for each parameter in the list
        for param in self.linearOperators:
            param.register_hook(self.scaleGradients)

    @staticmethod
    def scaleGradients(grad):
        return grad * 0.1

    def forward(self, inputData):
        # Cast the stability term to the device.
        self.restrictedWindowMask = self.restrictedWindowMask.to(inputData.device)
        self.stabilityTerm = self.stabilityTerm.to(inputData.device)

        for layerInd in range(self.numLayers):
            if not self.forwardDirection:
                pseudoLayerInd = self.numLayers - layerInd - 1
                inputData = self.applyLayer(inputData, pseudoLayerInd)
                inputData = self.activationFunctions[pseudoLayerInd](inputData)
            else:
                inputData = self.activationFunctions[layerInd](inputData)
                inputData = self.applyLayer(inputData, layerInd)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Get the current neural weights information.
        neuralWeights = self.restrictedWindowMask.clone()
        kernelWeights = self.linearOperators[layerInd]
        # neuralWeight: numSignals, sequenceLength, sequenceLength
        # kernelWeights: numSignals, (kernelSize) or (sequenceLength, sequenceLength)

        # Gather the corresponding kernel values for each position
        neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernelWeights[self.signalInds, self.kernelInds]

        # Add a stability term to the diagonal.
        neuralWeights = neuralWeights + self.stabilityTerm*0.975

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
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256
    _activationMethod = 'nonLinearMultiplication'
    _kernelSize = 3
    _numLayers = 5

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod, switchActivationDirection=False)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
