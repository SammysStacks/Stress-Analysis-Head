import math

import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleDualConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod, switchActivationDirection):
        super(reversibleDualConvolutionLayer, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.bounds = 1 / math.sqrt(kernelSize)  # The bounds for the neural weights.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.stabilityFactor1, self.stabilityFactor2 = 0.5, 0.5

        # The restricted window for the neural weights.
        self.restrictedWindowMask = torch.ones(1, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        self.restrictedWindowMask = torch.tril(torch.triu(self.restrictedWindowMask, diagonal=-kernelSize//2 + 1), diagonal=kernelSize//2)
        self.restrictedWindowMask = self.restrictedWindowMask.repeat(repeats=(numSignals, 1, 1))  # Dim: 1, sequenceLength, sequenceLength
        assert kernelSize <= sequenceLength, f"The kernel size is larger than the sequence length: {kernelSize}, {sequenceLength}"

        # Calculate the offsets to map positions to kernel indices
        self.signalInds, self.rowInds, self.colInds = self.restrictedWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.colInds - self.rowInds + self.kernelSize // 2  # Adjust for kernel center

        # Initialize the neural layers.
        self.activationFunctions1,  self.linearOperators1 = nn.ModuleList(), nn.ParameterList()
        self.activationFunctions2,  self.linearOperators2 = nn.ModuleList(), nn.ParameterList()

        # Create the neural layers.
        self.createArchitecture(self.linearOperators1, self.activationFunctions1, switchActivationDirection)
        self.createArchitecture(self.linearOperators2, self.activationFunctions2, switchActivationDirection)

        # Register hooks for each parameter in the list
        for param in self.linearOperators1: param.register_hook(self.scaleNeuralWeights)
        for param in self.linearOperators2: param.register_hook(self.scaleNeuralWeights)

    def createArchitecture(self, linearOperators, _activationFunctions, switchActivationDirection):
        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(self.numSignals, self.kernelSize, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds)
            linearOperators.append(parameters)

            # Add the activation function.
            activationMethod = f"{self.activationMethod}_{switchActivationDirection}"
            _activationFunctions.append(activationFunctions.getActivationMethod(activationMethod))
            switchActivationDirection = not switchActivationDirection

    def forward(self, inputData1, inputData2):
        # Cast the stability term to the device.
        self.restrictedWindowMask = self.restrictedWindowMask.to(inputData1.device)

        for layerInd in range(self.numLayers):
            if not self.forwardDirection:
                pseudoLayerInd = self.numLayers - layerInd - 1
                inputData1, inputData2 = self.applyLayer(inputData1, inputData2, pseudoLayerInd)
                inputData1 = self.activationFunctions1[pseudoLayerInd](inputData1)
                inputData2 = self.activationFunctions2[pseudoLayerInd](inputData2)
            else:
                inputData1 = self.activationFunctions1[layerInd](inputData1)
                inputData2 = self.activationFunctions2[layerInd](inputData2)
                inputData1, inputData2 = self.applyLayer(inputData1, inputData2, layerInd)

        return inputData1, inputData2

    def applyLayer(self, x1, x2, layerInd):
        # Get the current neural weights information.
        A1, A2 = self.restrictedWindowMask.clone(), self.restrictedWindowMask.clone()
        K1, K2 = self.linearOperators1[layerInd], self.linearOperators2[layerInd]
        # neuralWeight: numSignals, sequenceLength, sequenceLength
        # kernelWeights: numSignals, (kernelSize) or (sequenceLength, sequenceLength)

        # Gather the corresponding kernel values for each position
        A1[self.signalInds, self.rowInds, self.colInds] = K1[self.signalInds, self.kernelInds]
        A2[self.signalInds, self.rowInds, self.colInds] = K2[self.signalInds, self.kernelInds]

        # Mask the weights to match the kernel size.
        if self.kernelSize != self.sequenceLength: A1, A2 = self.restrictedWindowMask * A1, self.restrictedWindowMask * A2
        # A: numSignals, sequenceLength, sequenceLength

        if not self.forwardDirection:
            y1 = x1 + torch.einsum('bns,nsi->bni', x2, A1) * self.stabilityFactor1
            y2 = x2 + torch.einsum('bns,nsi->bni', y1, A2) * self.stabilityFactor2
        else:
            y2 = x2 - torch.einsum('bns,nsi->bni', x1, A2) * self.stabilityFactor2
            y1 = x1 - torch.einsum('bns,nsi->bni', y2, A1) * self.stabilityFactor1

        return y1, y2


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 128
    _activationMethod = 'nonLinearMultiplication'
    _kernelSize = 11
    _numLayers = 1

    # Set up the parameters.
    neuralLayerClass = reversibleDualConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod, switchActivationDirection=False)
    _inputData1 = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
    _inputData2 = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData1, _forwardData2, _reconstructedData1, _reconstructedData2 = neuralLayerClass.checkDualReconstruction(_inputData1, _inputData2, atol=1e-6, numLayers=1)
