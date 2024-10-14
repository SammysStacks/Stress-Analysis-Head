# General
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
# Import machine learning files
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralOperatorWeights, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                           skipConnectionProtocol, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, learningProtocol)
        # Initialize wavelet neural operator parameters.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the neural operator.
        self.highFrequenciesWeights = self.getHighFrequencyWeights()  # Learnable parameters for the high-frequency signal.
        self.lowFrequencyWeights = self.getLowFrequencyWeights()  # Learnable parameters for the low-frequency signal.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)  # Activation function for the Fourier neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the Fourier neural operator.

    def getSkipConnectionProtocol(self, skipConnectionProtocol):
        # Decide on the skip connection protocol.
        if skipConnectionProtocol == 'none':
            skipConnectionModel = self.zeros
        elif skipConnectionProtocol == 'identity':
            skipConnectionModel = nn.Identity()
        else: raise ValueError("The skip connection protocol must be in ['none', 'identity'].")

        return skipConnectionModel

    def getHighFrequencyWeights(self):
        highFrequenciesWeights = None
        if self.encodeHighFrequencies:

            # For each high frequency term.
            highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                highFrequencyParam = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd],
                                                                    finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd])
                # Store the high-frequency weights.
                highFrequenciesWeights.append(highFrequencyParam)

        return highFrequenciesWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        if self.encodeLowFrequency: lowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape, finalFrequencyDim=self.lowFrequencyShape)
        else: lowFrequencyWeights = None

        return lowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, outChannel, initialFrequencyDim, finalFrequencyDim):
        if self.learningProtocol == 'FCC': return self.neuralWeightFCC(inChannel=inChannel, outChannel=outChannel, finalFrequencyDim=finalFrequencyDim)
        elif self.learningProtocol == 'iCNN': return self.neuralWeightCNN(inChannel=inChannel, outChannel=inChannel, groups=inChannel)
        elif self.learningProtocol == 'CNN': return self.neuralWeightCNN(inChannel=inChannel, outChannel=outChannel, groups=1)
        elif self.learningProtocol == 'FC': return self.neuralWeightFC(numInputFeatures=initialFrequencyDim)
        else: raise ValueError(f"The learning protocol ({self.learningProtocol}) must be in ['FC', 'FCC', 'iCNN', 'CNN'].")

    @staticmethod
    def zeros(x): return 0
