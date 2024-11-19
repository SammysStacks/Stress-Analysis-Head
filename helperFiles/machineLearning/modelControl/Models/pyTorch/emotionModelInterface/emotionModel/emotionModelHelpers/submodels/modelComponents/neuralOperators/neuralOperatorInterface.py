from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.fourierOperator.fourierNeuralOperatorLayer import fourierNeuralOperatorLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights


class neuralOperatorInterface(emotionModelWeights):

    def __init__(self, operatorType, sequenceLength, numInputSignals, numOutputSignals, addBiasTerm):
        super().__init__()
        # General parameters.
        self.numOutputSignals = numOutputSignals  # The number of output signals.
        self.numInputSignals = numInputSignals  # The number of input signals.
        self.sequenceLength = sequenceLength  # The length of the input signals.
        self.operatorType = operatorType  # The type of operator to use.
        self.addBiasTerm = addBiasTerm  # Whether to add a bias term to the neural operator.

    def getNeuralOperatorLayer(self, neuralOperatorParameters, reversibleFlag):
        # Decide on the neural operator layer.
        if self.operatorType == 'wavelet': return self.initializeWaveletLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag, layerMultiple=1)
        elif self.operatorType == 'fourier': return self.initializeFourierLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag, layerMultiple=1)
        else: raise ValueError(f"The operator type ({self.operatorType}) must be in ['wavelet'].")

    def initializeWaveletLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag, layerMultiple):
        # Unpack the neural operator parameters.
        encodeHighFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeHighFrequencyProtocol', 'highFreq')  # The protocol for encoding the high frequency signals.
        encodeLowFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeLowFrequencyProtocol', 'lowFreq')  # The protocol for encoding the low frequency signals.
        waveletType = neuralOperatorParameters['wavelet'].get('waveletType', 'bior3.7')  # The type of wavelet to use for the wavelet transform.

        # Hardcoded parameters.
        mode = 'periodization'  # Mode: 'zero' (lossy), 'symmetric' (lossy), 'reflect' (lossy), or 'periodization' (lossless).
        numDecompositions = 1  # Hard to learn through more than one decomposition.

        # Hardcoded parameters.
        learningProtocol = 'rCNN' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        activationMethod = 'none' if reversibleFlag else emotionModelWeights.getIrreversibleActivation()  # The activation method to use.
        # numDecompositions = waveletNeuralOperatorLayer.max_decompositions(signal_length=sequenceLength, wavelet_name=waveletType) if learningProtocol not in ['drCNN', 'drFC'] else 1  # Number of decompositions for the waveletType transform.

        return waveletNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals*layerMultiple, numOutputSignals=self.numOutputSignals*layerMultiple, numDecompositions=numDecompositions,
                                          waveletType=waveletType, mode=mode, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                          encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol)

    def initializeFourierLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag, layerMultiple):
        # Unpack the neural operator parameters.
        encodeImaginaryFrequencies = neuralOperatorParameters.get('encodeImaginaryFrequencies', True)
        encodeRealFrequencies = neuralOperatorParameters.get('encodeRealFrequencies', True)

        # Hardcoded parameters.
        learningProtocol = 'rFC' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        activationMethod = 'none' if reversibleFlag else emotionModelWeights.getIrreversibleActivation()  # The activation method to use.

        return fourierNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals*layerMultiple, numOutputSignals=self.numOutputSignals*layerMultiple, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod,
                                          skipConnectionProtocol=skipConnectionProtocol, encodeRealFrequencies=encodeRealFrequencies, encodeImaginaryFrequencies=encodeImaginaryFrequencies, learningProtocol=learningProtocol)
