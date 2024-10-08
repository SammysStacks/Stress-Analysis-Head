# General
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class neuralOperatorInterface:

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, activationMethod, addBiasTerm):
        # General parameters.
        self.activationMethod = activationMethod
        self.numOutputSignals = numOutputSignals
        self.numInputSignals = numInputSignals
        self.sequenceLength = sequenceLength
        self.addBiasTerm = addBiasTerm

    def initializeNeuralLayer(self, neuralOperatorParameters):
        # Unpack the neural operator parameters.
        encodeHighFrequencyProtocol = neuralOperatorParameters['encodeHighFrequencyProtocol']
        encodeLowFrequencyProtocol = neuralOperatorParameters['encodeLowFrequencyProtocol']
        skipConnectionProtocol = neuralOperatorParameters['skipConnectionProtocol']
        independentChannels = neuralOperatorParameters['independentChannels']
        numDecompositions = neuralOperatorParameters['numDecompositions']
        learningProtocol = neuralOperatorParameters['learningProtocol']
        waveletType = neuralOperatorParameters['waveletType']
        mode = neuralOperatorParameters['mode']

        return waveletNeuralOperatorLayer(sequenceLength=self.sequenceLength, numInputSignals=self.numInputSignals, numOutputSignals=self.numOutputSignals, numDecompositions=numDecompositions,
                                          waveletType=waveletType, mode=mode, addBiasTerm=self.addBiasTerm, activationMethod=self.activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                          encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol, independentChannels=independentChannels)
