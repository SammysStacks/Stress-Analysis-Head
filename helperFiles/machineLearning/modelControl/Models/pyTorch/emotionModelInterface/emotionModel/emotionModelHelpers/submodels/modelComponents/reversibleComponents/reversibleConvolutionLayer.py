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
        self.numSignals = numSignals  # The number of signals in the input data.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.bounds = 1  # The bounds for the neural weights: lower values are like identity.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        if self.sequenceLength != self.kernelSize: upperWindowMask = torch.tril(upperWindowMask, diagonal=kernelSize//2)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.rowInds - self.colInds + self.kernelSize // 2  # Adjust for kernel center

        # Assert the validity of the input parameters.
        assert kernelSize <= sequenceLength - 1, f"The kernel size must be less than the sequence length: {kernelSize}, {sequenceLength}"
        assert self.kernelInds.max() == self.kernelSize//2 - 1, f"The kernel indices are not valid: {self.kernelInds.max()}"
        assert self.kernelInds.min() == 0, f"The kernel indices are not valid: {self.kernelInds.min()}"

        # Initialize the neural layers.
        self.activationFunctions,  self.linearOperators = nn.ModuleList(), nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(numSignals, kernelSize//2 or 1, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds)
            self.linearOperators.append(parameters)

            # Add the activation function.
            activationMethod = f"{self.activationMethod}_{switchActivationDirection}"
            self.activationFunctions.append(activationFunctions.getActivationMethod(activationMethod))
            switchActivationDirection = not switchActivationDirection

        # Scale the gradients.
        for layerInd in range(self.numLayers): self.linearOperators[layerInd].register_hook(self.scaleNeuralWeights)

    def forward(self, inputData):
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
        neuralWeights = torch.zeros(numSignals, sequenceLength, sequenceLength, dtype=torch.float64, device=inputData.device)
        # neuralWeight: numSignals, sequenceLength, sequenceLength

        # Assert the validity of the input parameters.
        assert sequenceLength == self.sequenceLength, f"The sequence length is not correct: {sequenceLength}, {self.sequenceLength}"
        assert numSignals == self.numSignals, f"The number of signals is not correct: {numSignals}, {self.numSignals}"

        # Gather the corresponding kernel values for each position for a skewed symmetric matrix.
        neuralWeights[:, self.rowInds, self.colInds] = -self.linearOperators[layerInd][:, self.kernelInds]
        neuralWeights[:, self.colInds, self.rowInds] = self.linearOperators[layerInd][:, self.kernelInds]
        # neuralWeight: numSignals, sequenceLength, sequenceLength

        # Create an orthogonal matrix.
        neuralWeights = neuralWeights.matrix_exp()
        if not self.forwardDirection: neuralWeights = neuralWeights.transpose(-2, -1)  # Ensure the neural weights are symmetric.
        # For orthogonal matrices: A.exp().inverse() = A.exp().transpose() = (-A).exp()

        # Apply the neural weights to the input data.
        outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)

        return outputData

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad) / self.numSignals
        print(f'The model has {numParams} trainable parameters.')


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 128
    _activationMethod = 'reversibleLinearSoftSign'
    _kernelSize = 31
    _numLayers = 1

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod, switchActivationDirection=False)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
    neuralLayerClass.printParams()
