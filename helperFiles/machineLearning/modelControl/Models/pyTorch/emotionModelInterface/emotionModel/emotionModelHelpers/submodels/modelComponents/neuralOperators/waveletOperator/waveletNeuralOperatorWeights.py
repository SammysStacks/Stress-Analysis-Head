from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numLayers, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralOperatorWeights, self).__init__(sequenceLength=sequenceLength, numInputSignals=numInputSignals, numOutputSignals=numOutputSignals, numLayers=numLayers, numDecompositions=numDecompositions,
                                                           waveletType=waveletType, mode=mode, addBiasTerm=addBiasTerm, activationMethod=activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                                           encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol)
        # Initialize wavelet neural operator parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)  # Activation function for the neural operator.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the neural operator.
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
            skipConnectionModel = self.skipConnectionFC(sequenceLength=self.expectedSequenceLength)
        else: raise ValueError("The skip connection protocol must be in ['none', 'identity', 'CNN'].")

        return skipConnectionModel

    def getHighFrequencyWeights(self):
        highFrequenciesWeights = None
        if self.encodeHighFrequencies:

            # For each high frequency term.
            highFrequenciesWeights = nn.ModuleList()
            numFrequencies = len(self.highFrequenciesShapes)
            for highFrequenciesInd in range(numFrequencies):
                if self.culledHighFrequencyBounds[0] <= highFrequenciesInd < self.culledHighFrequencyBounds[1]: highFrequencyParam = nn.Identity()
                else: highFrequencyParam = self.getNeuralWeightParameters(inChannel=self.numInputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd])
                highFrequenciesWeights.append(highFrequencyParam)

        return highFrequenciesWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        if self.encodeLowFrequency:
            lowFrequencyWeights = nn.ModuleList()
            lowFrequencyWeights.append(self.getNeuralWeightParameters(inChannel=self.numInputSignals, initialFrequencyDim=self.lowFrequencyShape))
        else: lowFrequencyWeights = None

        return lowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, initialFrequencyDim):
        if self.learningProtocol == 'rCNN': return self.reversibleNeuralWeightRCNN(numSignals=inChannel, sequenceLength=initialFrequencyDim, numLayers=self.numLayers)
        elif self.learningProtocol == 'FC': return self.neuralWeightFC(sequenceLength=initialFrequencyDim)
        else: raise ValueError(f"The learning protocol ({self.learningProtocol}) must be in ['FCC', 'rCNN', 'CNN'].")
