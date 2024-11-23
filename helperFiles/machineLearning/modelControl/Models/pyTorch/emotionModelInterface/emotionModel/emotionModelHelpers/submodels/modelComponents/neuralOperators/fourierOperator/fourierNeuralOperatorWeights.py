from torch import nn
import torch
import math

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions


class fourierNeuralOperatorWeights(emotionModelWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol):
        super(fourierNeuralOperatorWeights, self).__init__()
        # Fourier neural operator parameters.
        self.encodeImaginaryFrequencies = encodeImaginaryFrequencies  # Whether to encode the imaginary frequencies.
        self.skipConnectionProtocol = skipConnectionProtocol  # The skip connection protocol to use.
        self.encodeRealFrequencies = encodeRealFrequencies  # Whether to encode the real frequencies.
        self.expectedSequenceLength = sequenceLength  # The expected sequence length.
        self.activationMethod = activationMethod  # The activation method to use.
        self.learningProtocol = learningProtocol  # The learning protocol to use.
        self.numOutputSignals = numOutputSignals  # The number of output signals.
        self.numInputSignals = numInputSignals  # The number of input signals.
        self.addBiasTerm = addBiasTerm  # Whether to add bias terms to the output.

        # Set the sequence length to the next power of 2.
        self.sequenceLength = 2 ** int(torch.tensor(sequenceLength).log2().ceil())  # The length of the input signals.
        self.fourierDimension = self.sequenceLength // 2 + 1  # Number of Fourier modes (frequencies) to use.
        self.sequenceTimeWindow = None  # The time window for the sequence: Not yet implemented.

        # Initialize wavelet neural operator parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)  # Activation function for the Fourier neural operator.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the Fourier neural operator.
        if self.encodeImaginaryFrequencies: self.imaginaryFourierWeights = self.getNeuralWeightParameters(numInputSignals, self.fourierDimension)  # Learnable parameters for the low-frequency signal.
        if self.encodeRealFrequencies: self.realFourierWeights = self.getNeuralWeightParameters(numInputSignals, self.fourierDimension)  # Learnable parameters for the high-frequency signal.

        # Assert that the parameters are valid.
        self.assertValidParams()

    def assertValidParams(self):
        # Assert that the frequency protocol is valid.
        assert self.learningProtocol in ['FCC', 'rCNN', 'CNN', 'FC'], "Invalid learning protocol. Must be in ['FC', 'FCC', 'rCNN']."
        assert self.numInputSignals == self.numOutputSignals, "The number of input signals must equal the output signals for now."

    def getSkipConnectionProtocol(self, skipConnectionProtocol):
        # Decide on the skip connection protocol.
        if skipConnectionProtocol == 'none':
            skipConnectionModel = None
        elif skipConnectionProtocol == 'identity':
            skipConnectionModel = nn.Identity()
        elif skipConnectionProtocol == 'CNN':
            skipConnectionModel = self.skipConnectionCNN(numSignals=self.numInputSignals)
        elif skipConnectionProtocol == 'FC':
            skipConnectionModel = self.skipConnectionFC(sequenceLength=self.sequenceLength)
        else: raise ValueError("The skip connection protocol must be in ['none', 'identity', 'CNN'].")

        return skipConnectionModel

    def getNeuralWeightParameters(self, inChannel, fourierDimension):
        if self.learningProtocol == 'rCNN': return self.reversibleNeuralWeightRCNN(numSignals=inChannel, sequenceLength=fourierDimension)
        elif self.learningProtocol == 'FC': return self.neuralWeightFC(sequenceLength=fourierDimension)
        else: raise ValueError(f"The learning protocol ({self.learningProtocol}) must be in ['FCC', 'rCNN', 'CNN'].")

    def forwardFFT(self, inputData):
        # Perform the forward FFT and extract the magnitude and phase.
        fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='ortho')
        imaginaryFourierData = fourierData.imag
        realFourierData = fourierData.real

        return realFourierData, imaginaryFourierData

    def backwardFFT(self, realFourierData, imaginaryFourierData, resampledTimes=None):
        # Reconstruct the fourier data from the real and imaginary components.
        fourierData = realFourierData + 1j * imaginaryFourierData
        # fourierData: batchSize, numSignals, fourierDimension

        # Reconstruct the data based on the physiological times.
        if resampledTimes is None: return torch.fft.irfft(fourierData, n=self.sequenceLength, dim=-1, norm='ortho')
        print("Resampling the data: Review the code as it will not be perfect interpolation.")

        # Prepare the information for resampling.
        batchSize, numSignals, maxSequenceLength = resampledTimes.size()
        imagAngularFrequencies = self.getAngularFrequencies().to(realFourierData.device)
        fullFourierData = fourierData.view(batchSize, numSignals, 1, self.fourierDimension)
        # angularFrequencies: 1, 1, 1, fourierDimension

        # Reconstruct the data based on the new sampled times.
        basisFunctions = torch.exp(imagAngularFrequencies * resampledTimes.flip(dims=[-1]).view(batchSize, numSignals, maxSequenceLength, 1))
        reconstructedData = torch.sum(fullFourierData * basisFunctions, dim=-1) * 2 / math.sqrt(self.sequenceLength)
        # basisFunctions: batchSize, numSignals, maxSequenceLength, fourierDimension
        # fullFourierData: batchSize, numSignals, 1, fourierDimension
        # reconstructedData: batchSize, numSignals, maxSequenceLength

        return reconstructedData.real

    # SPECIFIC CASE. USE WITH CAUTION.
    def getAngularFrequencies(self):
        # Initialize the frequency indices for the fourier data.
        imagAngularFrequencies = 2j * torch.pi * torch.fft.rfftfreq(self.sequenceLength, d=self.sequenceTimeWindow / (self.sequenceLength - 1), dtype=torch.float64)
        print("Calculated frequencies are based on the sequence being evenly spaced from 0 -> sequenceTimeWindow")
        # angularFrequencies: fourierDimension

        return imagAngularFrequencies
