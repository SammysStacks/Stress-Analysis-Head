import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleLinearLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod, switchActivationDirection):
        super(reversibleLinearLayer, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.bounds = 1 / kernelSize  # The bounds for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # The stability term to add to the diagonal.
        self.stabilityTerm = torch.eye(self.sequenceLength, dtype=torch.float64)*0.99

        # The restricted window for the neural weights.
        self.restrictedWindowMask = torch.ones(1, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        if self.sequenceLength != self.kernelSize: self.restrictedWindowMask = torch.tril(torch.triu(self.restrictedWindowMask, diagonal=-kernelSize//2 + 1), diagonal=kernelSize//2)

        # Initialize the neural layers.
        self.activationFunctions,  self.linearOperators = nn.ModuleList(), nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(numSignals, sequenceLength, sequenceLength, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds) * self.restrictedWindowMask
            self.linearOperators.append(parameters)

            # Add the activation function.
            activationMethod = f"{self.activationMethod}_{switchActivationDirection}"
            self.activationFunctions.append(activationFunctions.getActivationMethod(activationMethod))
            switchActivationDirection = not switchActivationDirection

        # Scale the gradients.
        for layerInd in range(self.numLayers): self.linearOperators[layerInd].register_hook(self.scaleNeuralWeights)

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
        # Unpack the dimensions.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert sequenceLength == self.sequenceLength, f"The sequence length is not correct: {sequenceLength}, {self.sequenceLength}"

        # Apply a mask to the neural weights.
        neuralWeights = self.linearOperators[layerInd]
        # neuralWeight: numSignals, sequenceLength, sequenceLength

        # Add a stability term to the diagonal. TODO: Add sparse matrix support.
        if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        else: neuralWeights = neuralWeights + self.stabilityTerm

        # Backward direction: invert the neural weights.
        if not self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Apply the neural weights to the input data.
        outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)

        return outputData


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256
    _activationMethod = 'reversibleActivation'
    _kernelSize = 21
    _numLayers = 100

    # Set up the parameters.
    neuralLayerClass = reversibleLinearLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod, switchActivationDirection=False)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
