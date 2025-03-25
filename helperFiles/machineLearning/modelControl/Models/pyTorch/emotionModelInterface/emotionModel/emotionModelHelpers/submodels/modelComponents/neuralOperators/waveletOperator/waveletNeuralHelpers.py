# General
import pywt
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


# Notes:
# - The wavelet neural operator layer is a custom module that applies a wavelet decomposition and reconstruction to the input data.
# - The wavelet neural operator layer is used to learn the encoding of the input data.

# Wavelet options:
#   Biorthogonal Wavelets ('bior'):
#       bior1.1, bior1.3, bior1.5
#       bior2.2, bior2.4, bior2.6, bior2.8
#       bior3.1, bior3.3, bior3.5, bior3.7, bior3.9
#       bior4.4
#       bior5.5
#       bior6.8

#   Reverse Biorthogonal Wavelets:
#       rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8

#   Complex Gaussian Wavelets ('cgau'): COMPLEX CONTINUOUS
#       cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
#       cmor

#   Coiflet Wavelets ('coif'):
#       coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17

#   Daubechies Wavelets ('db'):
#       db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38

#   Gaussian Wavelets: CONTINUOUS
#       gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8

#   Symlet Wavelets:
#       sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20

#   Miscellaneous Wavelets and Other Families:
#       haar, dmey,  # Discrete Wavelets
#       fbsp, mexh, morl, shan  # Continuous Wavelets


class waveletNeuralHelpers(emotionModelWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numLayers, minWaveletDim, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralHelpers, self).__init__()
        # Fourier neural operator parameters.
        self.sequenceLength = 2 ** int(torch.tensor(sequenceLength).log2().ceil())  # The length of the input signals.
        self.encodeHighFrequencyProtocol = encodeHighFrequencyProtocol  # The high-frequency encoding protocol to use.
        self.encodeLowFrequencyProtocol = encodeLowFrequencyProtocol  # The low-frequency encoding protocol to use.
        self.skipConnectionProtocol = skipConnectionProtocol  # The skip connection protocol to use.
        self.expectedSequenceLength = sequenceLength  # The expected length of the input signals.
        self.activationMethod = activationMethod  # The activation method to use.
        self.learningProtocol = learningProtocol  # The learning protocol to use.
        self.numOutputSignals = numOutputSignals  # Number of output signals.
        self.numInputSignals = numInputSignals  # Number of input signals.
        self.minWaveletDim = minWaveletDim  # The minimum signal length to use for the wavelet decomposition.
        self.addBiasTerm = addBiasTerm  # Whether to add bias terms to the output.
        self.waveletType = waveletType  # The wavelet to use for the decomposition. Options: 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
        self.numLayers = numLayers  # The number of layers to use for the decomposition.
        self.mode = mode  # The padding mode to use for the decomposition. Options: 'zero', 'symmetric', 'reflect' or 'periodization'.

        # Assert that the parameters are valid.
        self.assertValidParams()

        # Decide on the frequency encoding protocol.
        highFrequencyProtocolInfo = encodeHighFrequencyProtocol.split("-")  # ['highFreq', 'numHighFreq2Learn']
        self.encodeHighFrequencies = highFrequencyProtocolInfo[0] in ['highFreq']  # Whether to encode the high frequencies.
        self.encodeLowFrequency = encodeLowFrequencyProtocol in ['lowFreq']  # Whether to encode the low-frequency signal.
        self.culledHighFrequencyBounds = (-1, -1) if len(highFrequencyProtocolInfo) == 1 else (int(highFrequencyProtocolInfo[1]), int(highFrequencyProtocolInfo[2]))

        # Initialize the wavelet decomposition and reconstruction layers.
        self.numDecompositions = self.max_decompositions(sequenceLength=self.sequenceLength, waveletType=self.waveletType, minWaveletDim=self.minWaveletDim)
        self.dwt = DWT1DForward(J=self.numDecompositions.item(), wave=self.waveletType, mode=self.mode)
        self.idwt = DWT1DInverse(wave=self.waveletType, mode=self.mode)

        # Check the final output sizes.
        self.lowFrequencyShape, self.highFrequenciesShapes = self.getWaveletDimensions(self.expectedSequenceLength)

    def assertValidParams(self):
        # Assert that the frequency protocol is valid.
        assert self.learningProtocol in ['FCC', 'rCNN', 'CNN', 'FC'], "Invalid learning protocol. Must be in ['FC', 'FCC', 'rCNN']."
        assert self.encodeHighFrequencyProtocol.split("-")[0] in ['highFreq', 'none'], "The high-frequency encoding protocol must be 'highFreq', 'none'."
        assert self.encodeLowFrequencyProtocol in ['lowFreq', 'none'], "The low-frequency encoding protocol must be 'lowFreq', 'none'."
        assert self.numInputSignals == self.numOutputSignals, "The number of input signals must equal the output signals for now."

    def max_decompositions(self, sequenceLength=None, waveletType=None, minWaveletDim=None):
        wavelet = pywt.Wavelet(waveletType or self.waveletType)
        filter_length = len(wavelet.dec_lo)  # Decomposition low-pass filter length
        filter_length = max(filter_length, minWaveletDim + 1 if minWaveletDim is not None else filter_length)
        max_level = torch.floor(torch.log2(torch.tensor((sequenceLength or self.sequenceLength) / (filter_length - 1), dtype=torch.float32))).int()
        return max_level

    def getWaveletDimensions(self, sequenceLength):
        # Get the expected output shapes (hard to calculate by hand).
        lowFrequency, highFrequencies = self.dwt(torch.randn(1, 1, sequenceLength))
        highFrequenciesShapes = [highFrequency.size(-1) for highFrequency in highFrequencies]  # Optimally: maxSequenceLength / decompositionLayer**2
        lowFrequencyShape = lowFrequency.size(-1)  # Optimally: maxSequenceLength / numDecompositions**2

        return lowFrequencyShape, highFrequenciesShapes
