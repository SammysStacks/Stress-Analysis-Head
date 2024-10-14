import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleLinearLayer(reversibleInterface):

    def __init__(self, sequenceLength, numLayers, activationMethod):
        super(reversibleLinearLayer, self).__init__()
        # General parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)  # The activation function to use for the neural operator.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # Initialize the neural weights.
        self.linearOperators = nn.ModuleList()
        for layerInd in range(self.numLayers):
            self.linearOperators.append(nn.Linear(sequenceLength, sequenceLength, bias=False))

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
        # Apply a mask to the neural weights.
        originalWeights = self.linearOperators[layerInd].weight.data.clone()
        firstNeuralWeights = torch.triu(originalWeights, diagonal=0)  # Causal weights
        secondNeuralWeights = torch.tril(originalWeights, diagonal=0)  # Non-causal weights
        # neuralWeight: outFeatures=sequenceLength, inFeatures=sequenceLength

        # Add a stability term to the diagonal.
        stabilityTerm = self.getStabilityTerm(self.sequenceLength, scalingFactor=1, device=inputData.device)
        firstNeuralWeights = firstNeuralWeights + stabilityTerm
        secondNeuralWeights = secondNeuralWeights - stabilityTerm
        # TODO: The stability term does not yet ensure convertibility.

        # Backward direction
        if not self.forwardDirection:
            # Invert the neural weights.
            firstNeuralWeights = torch.linalg.inv(firstNeuralWeights)
            secondNeuralWeights = torch.linalg.inv(secondNeuralWeights)

            # Swap the neural weights: A @ B -> A_inv @ B_inv
            secondNeuralWeights, firstNeuralWeights = firstNeuralWeights, secondNeuralWeights

        # Apply the neural weights to the input data.
        outputData = torch.matmul(inputData, firstNeuralWeights)
        outputData = torch.matmul(outputData, secondNeuralWeights)

        return outputData


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 4, 128
    _activationMethod = 'reversibleLinearSoftSign_1_0.9'
    _numLayers = 10

    # Set up the parameters.
    neuralLayerClass = reversibleLinearLayer(sequenceLength=_sequenceLength, numLayers=_numLayers, activationMethod=_activationMethod)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData)
