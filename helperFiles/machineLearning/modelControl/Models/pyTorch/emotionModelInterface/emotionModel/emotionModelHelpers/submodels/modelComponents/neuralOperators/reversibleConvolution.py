import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt


class reversibleConvolution(nn.Module):

    def __init__(self, numSignals, sequenceLength, kernelSize, independentChannels=False):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.sequenceLength = sequenceLength
        self.kernelSize = kernelSize
        self.numSignals = numSignals

        # Initialize a convolution kernel in the spatial domain
        self.kernel = torch.randn(numSignals, numSignals, kernelSize)
        if independentChannels: self.kernel = self.kernel * torch.eye(numSignals).unsqueeze(-1)
        self.kernel = nn.Parameter(self.kernel)  # output_channels, input_channels, kernel_size

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2

    def forward(self, inputData, forwardDirection):
        """ Perform multiplication in the Fourier domain (equivalent to spatial convolution). """
        # Pad the signals and kernel to N + K - 1 to maintain linear convolution.
        paddedKernel = nn.functional.pad(input=self.kernel, value=0, mode='constant', pad=(0, self.sequenceLength - self.kernelSize))

        # Compute Fourier transform of data and kernel.
        kernelFFT = torch.fft.rfft(paddedKernel, n=self.sequenceLength, dim=-1, norm='backward')
        fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='backward')

        if forwardDirection:
            # Multiply in the Fourier domain (equivalent to convolution in the spatial domain)
            convolutionalFourierData = torch.einsum('oif,bif->bof', kernelFFT, fourierData)
            # convolutionalFourierData = torch.zeros_like(fourierData)
            #
            # # Iterate over the batch dimension and the other dimensions manually
            # for batchInd in range(fourierData.size(0)):
            #     for inputChannelInd in range(kernelFFT.size(1)):
            #         for featureInd in range(fourierData.size(2)):
            #             # Perform element-wise multiplication along the 'i' dimension and sum it
            #             convolutionalFourierData[batchInd, inputChannelInd, featureInd] = fourierData[batchInd, inputChannelInd, featureInd] * kernelFFT[inputChannelInd, inputChannelInd, featureInd]

        else:
            # Perform einsum for the contracted operation as you did before
            convolutionalFourierData = torch.zeros_like(fourierData)
            kernelFFT = torch.linalg.pinv(kernelFFT.permute(2, 1, 0)).permute(2, 1, 0)

            # Iterate over the batch dimension and the other dimensions manually
            for batchInd in range(fourierData.size(0)):
                for inputChannelInd in range(kernelFFT.size(1)):
                    for featureInd in range(fourierData.size(2)):
                        # Perform element-wise multiplication along the 'i' dimension and sum it
                        convolutionalFourierData[batchInd, inputChannelInd, featureInd] = fourierData[batchInd, inputChannelInd, featureInd] * kernelFFT[inputChannelInd, inputChannelInd, featureInd]

        # Inverse Fourier transform to get back to spatial domain
        convolutionalData = torch.fft.irfft(convolutionalFourierData, n=self.sequenceLength, dim=-1, norm='backward')
        # convolutionalData = convolutionalData[:, :, self.padding:-self.padding - self.extraBuffer]

        return convolutionalData

    def checkEquivalence(self, inputData):
        batchSize, numSignals, sequenceLength = inputData.size()

        # Initialize a convolution kernel in the spatial domain
        conv1D = nn.Conv1d(in_channels=numSignals, out_channels=numSignals, kernel_size=self.kernelSize, stride=1,
                           padding=self.padding, dilation=1, groups=1, padding_mode='zeros', bias=False)
        conv1D.weight.data = torch.flip(self.kernel, dims=[-1]).data

        # Perform the convolution in the fourier and spatial domains.
        fourierConvolution = self.forward(inputData, forwardDirection=True)
        spatialConvolution = conv1D(inputData)
        print(fourierConvolution.size(), spatialConvolution.size())

        # Compare the two results
        if torch.allclose(spatialConvolution, fourierConvolution, atol=1e-5): print("The convolution in spatial and Fourier domains are equivalent!")
        else: print("There is a discrepancy between the two methods.")

        # Plot the results
        plt.plot(spatialConvolution[0][0].detach().numpy(), 'k', linewidth=1.25, label='Spatial Convolution')
        plt.plot(fourierConvolution[0][0].detach().numpy(), 'tab:red', linewidth=1, label='Spatial Convolution')
        plt.show()

        return spatialConvolution, fourierConvolution

    def checkReconstruction(self, inputData):
        # Perform the forward/backward convolutions.
        fourierConvolution = self.forward(inputData, forwardDirection=True)
        reconstructedData = self.forward(fourierConvolution, forwardDirection=False)

        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=1e-5): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(inputData[0][0].detach().numpy(), 'k', linewidth=1.25, label='Initial Signal')
        plt.plot(reconstructedData[0][0].detach().numpy(), 'tab:red', linewidth=1, label='Reconstructed Signal')
        plt.legend()
        plt.show()

        return inputData, reconstructedData


if __name__ == "__main__":
    # General parameters.
    batchSizeTemp, numSignalsTemp, sequenceLengthTemp = 2, 4, 128
    independentChannelsTemp = True
    kernelSizeTemp = 13

    # Set up the parameters.
    convolutionalClass = reversibleConvolution(numSignalsTemp, sequenceLengthTemp, kernelSizeTemp, independentChannelsTemp)
    inputDataTemp = torch.randn(batchSizeTemp, numSignalsTemp, sequenceLengthTemp)

    # Perform the convolution in the fourier and spatial domains.
    spatialConvolutionTemp, fourierConvolutionTemp = convolutionalClass.checkEquivalence(inputDataTemp)
    # outputDataTemp, reconstructedDataTemp = convolutionalClass.checkReconstruction(inputDataTemp)
