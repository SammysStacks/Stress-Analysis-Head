import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolution(reversibleInterface):

    def __init__(self, numChannels, kernelSize):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.numChannels = numChannels
        self.kernelSize = kernelSize

        # Initialize the neural weights.
        self.linearOperator = nn.Conv1d(in_channels=numChannels, out_channels=numChannels, kernel_size=kernelSize, stride=1, padding=0, dilation=1, groups=numChannels, padding_mode='reflect', bias=False)
        self.stabilityTerm = torch.eye(self.kernelSize)*0.5

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2  # Assuming dilation = 1
        assert self.kernelSize % 2 == 1, "The kernel size must be odd."

    def forward(self, inputData):
        # Skip connection.
        skipConnectionWeight = torch.zeros(self.kernelSize, device=inputData.device)
        skipConnectionWeight[self.kernelSize//2] = 1

        # Create a square neural weight matrix.
        neuralWeights = self.linearOperator.weight.data + skipConnectionWeight
        neuralWeights = neuralWeights.unsqueeze(-1).repeat(1, 1, 1, self.kernelSize)
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Create an invertible neural weight matrix.
        neuralWeights = torch.triu(neuralWeights, diagonal=0) + self.stabilityTerm
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Invert the neural weights if needed.
        if not self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Perform the convolution.
        convolutionalData = torch.nn.functional.conv1d(inputData, neuralWeights[:, :, :, 0], bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numChannels)
        convolutionalData = convolutionalData[:, :, self.kernelSize:-1]

        return convolutionalData

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
    _kernelSize = 23

    # Set up the parameters.
    neuralLayerClass = reversibleConvolution(numChannels=_numSignals, kernelSize=_kernelSize)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData)
