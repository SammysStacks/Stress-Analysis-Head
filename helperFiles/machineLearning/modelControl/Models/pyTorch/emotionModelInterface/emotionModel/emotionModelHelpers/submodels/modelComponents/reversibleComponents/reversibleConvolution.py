import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolution(reversibleInterface):

    def __init__(self, numChannels, kernelSize, numLayers, skipConnection):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.skipConnection = skipConnection  # Whether to use a skip connection.
        self.numChannels = numChannels  # The number of channels in the input signal.
        self.kernelSize = kernelSize  # The size of the kernel for the convolution.
        self.numLayers = numLayers  # The number of layers in the reversible convolution.

        # Initialize the neural weights.
        self.linearOperators = nn.ModuleList()
        for layerInd in range(self.numLayers):
            self.linearOperators.append(nn.Conv1d(in_channels=numChannels, out_channels=numChannels, kernel_size=kernelSize, stride=1, padding=0, dilation=1, groups=numChannels, padding_mode='reflect', bias=False))

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2  # Assuming dilation = 1
        assert self.kernelSize % 2 == 1, "The kernel size must be odd."

    @staticmethod
    def getStabilityTerm(kernelSize):
        return torch.eye(kernelSize)*0.5

    def forward(self, inputData):
        # Prepare the neural weights.
        neuralWeights = self.combineNeuralWeights()
        kernelSize = neuralWeights.size(-1)

        # Skip connection.
        skipConnectionWeight = torch.zeros(kernelSize, device=inputData.device)
        skipConnectionWeight[kernelSize//2] = 1 if self.skipConnection else 0
        neuralWeights = neuralWeights + skipConnectionWeight

        # Create a square neural weight matrix.
        neuralWeights = neuralWeights.unsqueeze(-1).repeat(1, 1, 1, kernelSize)
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Create an invertible neural weight matrix.
        stabilityTerm = self.getStabilityTerm(kernelSize)
        neuralWeights = torch.triu(neuralWeights, diagonal=0) + stabilityTerm
        # numInputChannels, numOutputChannels=1, kernelSize, kernelSize

        # Invert the neural weights if needed.
        if not self.forwardDirection: neuralWeights = torch.linalg.inv(neuralWeights)

        # Perform the convolution.
        convolutionalData = torch.nn.functional.conv1d(inputData, neuralWeights[:, :, :, 0], bias=None, stride=1, padding=kernelSize, dilation=1, groups=self.numChannels)
        convolutionalData = convolutionalData[:, :, kernelSize:-1]

        return convolutionalData

    def combineNeuralWeights(self):
        # Extract the first set of weights.
        neuralWeights = self.linearOperators[0].weight.data.clone()

        # For each layer.
        for layerInd in range(self.numLayers):
            if layerInd == 0:
                neuralWeights = neuralWeights.permute(1, 0, 2)
            else:
                # Extract the next set of weights and flip them.
                nextWeights = self.linearOperators[layerInd].weight.data.clone()
                nextWeights = torch.flip(nextWeights, dims=[-1])  # Flip the kernel dimension.

                # Perform the convolution.
                neuralWeights = torch.nn.functional.conv1d(neuralWeights, nextWeights, bias=None, stride=1, padding=self.kernelSize, dilation=1, groups=self.numChannels)
                neuralWeights = neuralWeights[:, :, 1:-1]
        neuralWeights = neuralWeights.permute(1, 0, 2)

        return neuralWeights

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
    _kernelSize = 7
    _numLayers = 4

    # Set up the parameters.
    neuralLayerClass = reversibleConvolution(numChannels=_numSignals, kernelSize=_kernelSize, numLayers=_numLayers, skipConnection=True)
    _inputData = torch.randn(_batchSize, _numSignals, _sequenceLength)

    # Perform the convolution in the fourier and spatial domains.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(_inputData)
