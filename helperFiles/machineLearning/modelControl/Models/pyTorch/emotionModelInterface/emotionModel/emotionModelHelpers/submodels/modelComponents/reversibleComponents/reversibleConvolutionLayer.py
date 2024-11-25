import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        if self.sequenceLength != self.kernelSize: upperWindowMask = torch.tril(upperWindowMask, diagonal=self.kernelSize//2)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.rowInds - self.colInds + self.kernelSize // 2  # Adjust for the kernel center

        # Assert the validity of the input parameters.
        assert 1 <= self.kernelSize//2 <= sequenceLength - 1, f"The kernel size must be less than the sequence length: {self.kernelSize}, {self.sequenceLength}"
        assert self.kernelInds.max() == self.kernelSize//2 - 1, f"The kernel indices are not valid: {self.kernelInds.max()}"
        assert self.kernelInds.min() == 0, f"The kernel indices are not valid: {self.kernelInds.min()}"

        # Initialize the neural layers.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)
        self.linearOperators = nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(numSignals, self.kernelSize//2 or 1, dtype=torch.float64))
            parameters = nn.init.kaiming_normal_(parameters)
            self.linearOperators.append(parameters)

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if not self.forwardDirection: layerInd = self.numLayers - layerInd - 1

            # Apply the weights to the input data.
            if self.activationMethod == 'none': inputData = self.applyLayer(inputData, layerInd)
            else: inputData = self.activationFunction(inputData, lambda x: self.applyLayer(x, layerInd), forwardFirst=layerInd % 2 == 0)

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
        if self.forwardDirection: neuralWeights = neuralWeights.transpose(-2, -1)  # Ensure the neural weights are symmetric.
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
    _batchSize, _numSignals, _sequenceLength = 64, 128, 128
    _activationMethod = 'reversibleLinearSoftSign_2'
    _kernelSize = 2*_sequenceLength - 1
    _numLayers = 2

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
    _inputData = nn.init.normal_(_inputData, mean=0, std=1)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
    neuralLayerClass.printParams()
