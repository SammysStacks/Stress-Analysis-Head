import torch
import torch.nn as nn
import torch.fft


class reversibleConvolution(nn.Module):

    def __init__(self, batchSize, numSignals, sequenceLength, kernelSize, independentChannels=False):
        super(reversibleConvolution, self).__init__()
        # General parameters.
        self.sequenceLength = sequenceLength
        self.kernelSize = kernelSize
        
        # Initialize a convolution kernel in the spatial domain
        if independentChannels: kernel = torch.randn(1, 1, kernelSize).expand(batchSize, numSignals, kernelSize)
        else: kernel = torch.randn(1, numSignals, kernelSize).expand(batchSize, numSignals, kernelSize)
        self.kernel = nn.Parameter(kernel)
        print(self.kernel.size())

    def forward(self, inputData):
        """ Perform multiplication in the Fourier domain (equivalent to spatial convolution). """
        # Compute Fourier transform of input.
        fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='ortho')

        # Pad the kernel to the size of the input for Fourier multiplication
        padded_kernel = nn.functional.pad(input=self.kernel, value=0, mode='constant', pad=(0, self.sequenceLength - self.kernelSize))
        kernelFFT = torch.fft.rfft(padded_kernel, n=self.sequenceLength, dim=-1, norm='ortho')

        # Multiply in the Fourier domain (equivalent to convolution in the spatial domain)
        convolutionalData = kernelFFT * fourierData

        # Inverse Fourier transform to get back to spatial domain
        outputData = torch.fft.irfft(convolutionalData, n=self.sequenceLength, dim=-1, norm='ortho')

        return outputData

    def checkEquivalence(self, inputData):
        # Perform the convolution in the fourier and spatial domains.
        spatialConvolution = nn.functional.conv1d(inputData, weight=self.kernel, bias=None, padding='valid', stride=1, dilation=1, groups=1)
        fourierConvolution = self.forward(inputData)

        # Compare the two results
        if torch.allclose(spatialConvolution, fourierConvolution, atol=1e-5): print("The convolution in spatial and Fourier domains are equivalent!")
        else: print("There is a discrepancy between the two methods.")
        print("spatialConvolution", spatialConvolution[0][0])
        print("fourierConvolution", fourierConvolution[0][0])

        return spatialConvolution, fourierConvolution


if __name__ == "__main__":
    # General parameters.
    batchSizeTemp, numSignalsTemp, sequenceLengthTemp = 2, 2, 32
    kernelSizeTemp = 3
    
    # Set up the parameters.
    convolutionalClass = reversibleConvolution(batchSizeTemp, numSignalsTemp, sequenceLengthTemp, kernelSizeTemp)
    inputDataTemp = torch.randn(batchSizeTemp, numSignalsTemp, sequenceLengthTemp)

    # Perform the convolution in the fourier and spatial domains.
    spatialConvolutionTemp, fourierConvolutionTemp = convolutionalClass.checkEquivalence(inputDataTemp)
