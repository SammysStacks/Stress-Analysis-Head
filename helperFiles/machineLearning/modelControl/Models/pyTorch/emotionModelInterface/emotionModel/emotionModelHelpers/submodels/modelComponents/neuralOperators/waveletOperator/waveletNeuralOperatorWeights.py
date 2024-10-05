# General
from torch import nn

# Import machine learning files
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, learningProtocol=0, independentChannels=False):
        super(waveletNeuralOperatorWeights, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                           skipConnectionProtocol, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, learningProtocol, independentChannels)
        # Initialize wavelet neural operator parameters.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the neural operator.
        self.highFrequenciesWeights = self.getHighFrequencyWeights()  # Learnable parameters for the high-frequency signal.
        self.lowFrequencyWeights = self.getLowFrequencyWeights()  # Learnable parameters for the low-frequency signal.
        self.activationFunction = self.getActivationMethod(activationType=activationMethod)  # Activation function for the Fourier neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the Fourier neural operator.

    def getSkipConnectionProtocol(self, skipConnectionProtocol):
        # Decide on the skip connection protocol.
        if skipConnectionProtocol == 'identity':
            skipConnectionModel = nn.Identity()
        elif skipConnectionProtocol == 'singleCNN':
            skipConnectionModel = self.skipConnectionEncoding(inChannel=self.numInputSignals, outChannel=self.numOutputSignals)
        elif skipConnectionProtocol == 'independentCNN':
            skipConnectionModel = self.independentSkipConnectionEncoding(inChannel=self.numInputSignals, outChannel=self.numOutputSignals)
        else:
            raise ValueError("The skip connection protocol must be in ['none', 'identity', 'CNN'].")

        return skipConnectionModel

    def getHighFrequencyWeights(self):
        highFrequenciesWeights = None
        if self.encodeHighFrequencies:

            # For each high frequency term.
            highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                highFrequencyParam = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd],
                                                                    finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd], lowFreqSignal=False)
                # Store the high-frequency weights.
                highFrequenciesWeights.append(highFrequencyParam)

        return highFrequenciesWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        if self.encodeLowFrequency:
            lowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape, finalFrequencyDim=self.lowFrequencyShape, lowFreqSignal=True)
        else:
            lowFrequencyWeights = None

        return lowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, outChannel, initialFrequencyDim, finalFrequencyDim, lowFreqSignal=False):
        if self.useConvolutionFlag:
            # Initialize the independent frequency weights.
            if self.independentChannels: return self.independentNeuralWeightCNN(inChannel=inChannel, outChannel=outChannel)

            # Initialize the frequency weights.
            if lowFreqSignal:
                return self.neuralWeightLowCNN(inChannel=inChannel, outChannel=outChannel)
            else:
                return self.neuralWeightHighCNN(inChannel=inChannel, outChannel=outChannel)

        else:
            if self.independentChannels:
                # Initialize the high-frequency weights to learn how to change.
                assert inChannel == outChannel, "The number of input and output signals must be equal."
                return self.neuralWeightIndependentModel(numInputFeatures=initialFrequencyDim, numOutputFeatures=finalFrequencyDim)

            if initialFrequencyDim == finalFrequencyDim:
                # Initialize the high-frequency weights to learn how to change the channels.
                return self.neuralWeightParameters(inChannel=inChannel, outChannel=outChannel, finalFrequencyDim=finalFrequencyDim)
            else:
                # Initialize the high-frequency weights to learn how to change the channels.
                assert inChannel == outChannel, "The number of input and output signals must be equal."
                return self.neuralCombinationWeightParameters(inChannel=outChannel, initialFrequencyDim=initialFrequencyDim, finalFrequencyDim=finalFrequencyDim)
