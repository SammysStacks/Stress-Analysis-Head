import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleLinearLayer(reversibleInterface):

    def __init__(self, sequenceLength):
        super(reversibleLinearLayer, self).__init__()
        # General parameters.
        self.sequenceLength = sequenceLength

        # Initialize the neural weights.
        self.linearOperator = nn.Linear(sequenceLength, sequenceLength, bias=False)
        self.stabilityTerm = torch.eye(self.sequenceLength)

    def forward(self, inputData):
        # Apply a mask to the neural weights.
        firstNeuralWeights = torch.triu(self.linearOperator.weight.data, diagonal=0)  # Causal weights
        secondNeuralWeights = torch.tril(self.linearOperator.weight.data, diagonal=0)  # Non-causal weights
        # neuralWeight: outFeatures=sequenceLength, inFeatures=sequenceLength

        # Add a stability term to the diagonal.
        firstNeuralWeights = firstNeuralWeights + self.stabilityTerm
        secondNeuralWeights = secondNeuralWeights - self.stabilityTerm
        # TODO: The stability term does not yet ensure invertibility.

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

    def checkReconstruction(self, inputData):
        # Forward direction.
        self.forwardDirection = True
        forwardData = self.forward(inputData)

        # Backward direction.
        self.forwardDirection = False
        reconstructedData = self.forward(forwardData)

        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=1e-4): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(inputData[0][0].detach().numpy(), 'k', linewidth=1.5, label='Initial Signal')
        plt.plot(reconstructedData[0][0].detach().numpy(), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
        plt.plot(forwardData[0][0].detach().numpy(), 'tab:blue', linewidth=1, label='Latent Signal', alpha=0.5)
        plt.legend()
        plt.show()

        return forwardData, reconstructedData


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 4, 128

    # Set up the parameters.
    neuralLayerClass = reversibleLinearLayer(sequenceLength=_sequenceLength)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData)
