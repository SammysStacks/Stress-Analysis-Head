import numpy as np
import torch
import torch.nn as nn
import torch.fft
from matplotlib import pyplot as plt


class reversibleConvolution(nn.Module):

    def __init__(self, numSignals, sequenceLength, kernelSize, independentChannels=False):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.paddedSequenceLength = sequenceLength + kernelSize - 1
        self.sequenceLength = sequenceLength
        self.kernelSize = kernelSize

        # Initialize a convolution kernel in the spatial domain
        if independentChannels: kernel = torch.randn(1, 1, kernelSize).expand(numSignals, numSignals, kernelSize)
        else: kernel = torch.randn(1, numSignals, kernelSize).expand(numSignals, numSignals, kernelSize)
        self.kernel = nn.Parameter(kernel)

        # Calculate the padding for the convolution.
        self.padding = (kernelSize - 1) // 2

    def forward(self, inputData):
        """ Perform multiplication in the Fourier domain (equivalent to spatial convolution). """
        # Pad the signals and kernel to N + K - 1 to maintain linear convolution.
        paddedSignals = nn.functional.pad(input=inputData, value=0, mode='constant', pad=(0, self.paddedSequenceLength - self.sequenceLength))
        paddedKernel = nn.functional.pad(input=self.kernel, value=0, mode='constant', pad=(0, self.paddedSequenceLength - self.kernelSize))

        # Compute Fourier transform of data and kernel.
        fourierData = torch.fft.rfft(paddedSignals, n=self.paddedSequenceLength, dim=-1, norm='backward')
        kernelFFT = torch.fft.rfft(paddedKernel, n=self.paddedSequenceLength, dim=-1, norm='backward')

        # Multiply in the Fourier domain (equivalent to convolution in the spatial domain)
        convolutionalFourierData = torch.einsum('oif,bif->bof', kernelFFT, fourierData)

        # Inverse Fourier transform to get back to spatial domain
        convolutionalData = torch.fft.irfft(convolutionalFourierData, n=self.paddedSequenceLength, dim=-1, norm='backward')
        convolutionalData = convolutionalData[:, :, self.padding:-self.padding]

        return convolutionalData

    def inverse_forward(self, outputData):
        """ Reconstruct inputData from outputData using the same logic """
        # Pad the signals and kernel to N + K - 1 to maintain linear convolution.
        paddedKernel = nn.functional.pad(input=self.kernel, value=0, mode='constant', pad=(0, self.paddedSequenceLength - self.kernelSize))

        # Compute the Fourier transform of paddedOutputData to get convolutionalData
        convolutionalData = torch.fft.rfft(outputData, n=self.paddedSequenceLength, dim=-1, norm='backward')
        convolutionalData = convolutionalData.permute(0, 2, 1)  # Shape: [batchSize, freq_bins, numSignals]

        # Compute Fourier transform of the kernel
        kernelFFT = torch.fft.rfft(paddedKernel, n=self.paddedSequenceLength, dim=-1, norm='backward')
        kernelFFT = kernelFFT.permute(2, 0, 1)  # Shape: [freq_bins, out_channels, in_channels]

        # Compute the pseudo-inverse of kernelFFT at each frequency bin
        K = kernelFFT  # Shape: [freq_bins, out_channels, in_channels]
        K_inv = torch.linalg.pinv(K)  # Shape: [freq_bins, in_channels, out_channels]

        # Solve for fourierData using torch.einsum
        fourierData = torch.einsum('fio,bfo->bfi', K_inv, convolutionalData)
        fourierData = fourierData.permute(0, 2, 1)  # Shape: [batchSize, in_channels, freq_bins]

        # Compute the inverse Fourier transform to get paddedSignals
        reconstructedData = torch.fft.irfft(fourierData, n=self.paddedSequenceLength, dim=-1, norm='backward')
        # Shape: [batchSize, in_channels, self.paddedSequenceLength]

        return reconstructedData

    def checkEquivalence(self, inputData):
        batchSize, numSignals, sequenceLength = inputData.size()

        # Initialize a convolution kernel in the spatial domain
        conv1D = nn.Conv1d(in_channels=numSignals, out_channels=numSignals, kernel_size=self.kernelSize, stride=1,
                           padding=self.padding, dilation=1, groups=1, padding_mode='zeros', bias=False)
        conv1D.weight.data = torch.flip(self.kernel, dims=[-1]).data

        # Perform the convolution in the fourier and spatial domains.
        fourierConvolution = self.forward(inputData)
        spatialConvolution = conv1D(inputData)

        # Compare the two results
        if torch.allclose(spatialConvolution, fourierConvolution, atol=1e-5): print("The convolution in spatial and Fourier domains are equivalent!")
        else: print("There is a discrepancy between the two methods.")

        # Plot the results
        plt.plot(spatialConvolution[0][0].detach().numpy())
        plt.plot(fourierConvolution[0][0].detach().numpy())
        # plt.plot(inputData[0][0].detach().numpy(), 'k')
        plt.show()

        return spatialConvolution, fourierConvolution

    def checkReconstruction(self, inputData):
        # Perform forward convolution
        outputData = self.forward(inputData)
        reconstructedData = self.inverse_forward(outputData)

        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=1e-5): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(inputData[0][0].detach().numpy(), label='Original Input')
        plt.plot(reconstructedData[0][0].detach().numpy(), label='Reconstructed Input', linestyle='--')
        plt.legend()
        plt.show()

        return outputData, reconstructedData


if __name__ == "__main__":
    # General parameters.
    batchSizeTemp, numSignalsTemp, sequenceLengthTemp = 2, 4, 128
    independentChannelsTemp = False
    kernelSizeTemp = 3

    # Set up the parameters.
    convolutionalClass = reversibleConvolution(numSignalsTemp, sequenceLengthTemp, kernelSizeTemp, independentChannelsTemp)
    inputDataTemp = torch.randn(batchSizeTemp, numSignalsTemp, sequenceLengthTemp)

    # Perform the convolution in the fourier and spatial domains.
    spatialConvolutionTemp, fourierConvolutionTemp = convolutionalClass.checkEquivalence(inputDataTemp)
    # outputDataTemp, reconstructedDataTemp = convolutionalClass.checkReconstruction(inputDataTemp)
