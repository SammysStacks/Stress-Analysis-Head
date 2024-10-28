from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralOperatorWeights, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                           skipConnectionProtocol, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, learningProtocol)
        # Initialize wavelet neural operator parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)  # Activation function for the neural operator.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the neural operator.

        if self.learningProtocol in ['rFC', 'rCNN']:
            self.dualFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, initialFrequencyDim=self.lowFrequencyShape)  # Learnable parameters for the dual-frequency signal.
            assert numDecompositions == 1, f"The number of decompositions must be 1 for the dual-frequency signal: {numDecompositions}"
        else:
            self.highFrequenciesWeights = self.getHighFrequencyWeights()  # Learnable parameters for the high-frequency signal.
            self.lowFrequencyWeights = self.getLowFrequencyWeights()  # Learnable parameters for the low-frequency signal.

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

    def getHighFrequencyWeights(self):
        highFrequenciesWeights = None
        if self.encodeHighFrequencies:

            # For each high frequency term.
            highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                highFrequencyParam = self.getNeuralWeightParameters(inChannel=self.numInputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd])
                # Store the high-frequency weights.
                highFrequenciesWeights.append(highFrequencyParam)

        return highFrequenciesWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        if self.encodeLowFrequency: lowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, initialFrequencyDim=self.lowFrequencyShape)
        else: lowFrequencyWeights = None

        return lowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, initialFrequencyDim):
        if self.learningProtocol == 'rFC': return self.neuralWeightRFC(numSignals=inChannel, sequenceLength=initialFrequencyDim, activationMethod=self.activationMethod)
        elif self.learningProtocol == 'rCNN': return self.reversibleNeuralWeightRCNN(numSignals=inChannel, sequenceLength=initialFrequencyDim, activationMethod=self.activationMethod)
        elif self.learningProtocol == 'FC': return self.neuralWeightFC(sequenceLength=initialFrequencyDim)
        else: raise ValueError(f"The learning protocol ({self.learningProtocol}) must be in ['rFC', 'FCC', 'rCNN', 'CNN'].")
