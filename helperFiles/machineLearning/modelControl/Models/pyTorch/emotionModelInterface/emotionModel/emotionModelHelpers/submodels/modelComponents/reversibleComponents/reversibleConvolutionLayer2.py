import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer2(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod, switchActivationDirection):
        super(reversibleConvolutionLayer2, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.bounds = 1 / kernelSize  # The bounds for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # The stability term to add to the diagonal.
        self.stabilityTerm = torch.eye(self.sequenceLength, dtype=torch.float64)*0.99

        # The restricted window for the neural weights.
        restrictedLowerWindowMask = torch.ones(numSignals, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        restrictedLowerWindowMask = torch.tril(torch.triu(restrictedLowerWindowMask, diagonal=-kernelSize // 2 + 1), diagonal=0)
        # Calculate the offsets to map positions to kernel indices
        self.lowerSignalInds, self.lowerRowInds, self.lowerColInds = restrictedLowerWindowMask.nonzero(as_tuple=False).T
        self.lowerKernelInds = self.lowerColInds - self.lowerRowInds + kernelSize // 2  # Adjust for kernel center

        # The restricted window for the neural weights.
        restrictedUpperWindowMask = torch.ones(numSignals, self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        restrictedUpperWindowMask = torch.triu(torch.tril(restrictedUpperWindowMask, diagonal=kernelSize // 2), diagonal=0)
        # Calculate the offsets to map positions to kernel indices
        self.upperSignalInds, self.upperRowInds, self.upperColInds = restrictedUpperWindowMask.nonzero(as_tuple=False).T
        self.upperKernelInds = self.upperColInds - self.upperRowInds + kernelSize // 2 + 1  # Adjust for kernel center

        # Assert the validity of the input parameters.
        assert kernelSize <= sequenceLength, f"The kernel size is larger than the sequence length: {kernelSize}, {sequenceLength}"
        assert kernelSize % 2 == 1, f"The kernel size must be odd: {kernelSize}"

        # Initialize the neural layers.
        self.activationFunctions,  self.linearOperators = nn.ModuleList(), nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            parameters = nn.Parameter(torch.randn(numSignals, kernelSize + 1, dtype=torch.float64))
            self.linearOperators.append(nn.init.uniform_(parameters, a=-self.bounds, b=self.bounds))

            # Add the activation function.
            activationMethod = f"{self.activationMethod}_{switchActivationDirection}"
            self.activationFunctions.append(activationFunctions.getActivationMethod(activationMethod))
            switchActivationDirection = not switchActivationDirection

        # Scale the gradients.
        for layerInd in range(self.numLayers): self.linearOperators[layerInd].register_hook(self.scaleNeuralWeights)

    def forward(self, inputData):
        # Cast the stability term to the device.
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

        # Get the current neural weights information.
        kernelWeights = self.linearOperators[layerInd]
        lowerNeuralWeights = torch.zeros(numSignals, sequenceLength, sequenceLength, dtype=torch.float64, device=inputData.device)
        upperNeuralWeights = torch.zeros(numSignals, sequenceLength, sequenceLength, dtype=torch.float64, device=inputData.device)
        # neuralWeight: numSignals, sequenceLength, sequenceLength
        # kernelWeights: numSignals, kernelSize + 1

        # Gather the corresponding kernel values for each position
        lowerNeuralWeights[self.lowerSignalInds, self.lowerRowInds, self.lowerColInds] = kernelWeights[self.lowerSignalInds, self.lowerKernelInds]  # Lower triangular: future values.
        upperNeuralWeights[self.upperSignalInds, self.upperRowInds, self.upperColInds] = kernelWeights[self.upperSignalInds, self.upperKernelInds]  # Upper triangular: past values.

        # Add stability to the diagonal.
        lowerNeuralWeights = lowerNeuralWeights + self.stabilityTerm
        upperNeuralWeights = upperNeuralWeights + self.stabilityTerm

        # Backward direction: invert the neural weights.
        if not self.forwardDirection: upperNeuralWeights, lowerNeuralWeights = torch.linalg.inv(lowerNeuralWeights), torch.linalg.inv(upperNeuralWeights)

        # Apply the neural weights to the input data.
        outputData = torch.einsum('bns,nsi->bni', inputData, upperNeuralWeights)
        outputData = torch.einsum('bns,nsi->bni', outputData, lowerNeuralWeights)

        return outputData


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256
    _activationMethod = 'reversibleLinearSoftSign'
    _kernelSize = 5
    _numLayers = 1

    # Set up the parameters.
    neuralLayerClass = reversibleConvolutionLayer2(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod, switchActivationDirection=False)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData, atol=1e-6, numLayers=1)
