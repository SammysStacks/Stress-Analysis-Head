from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.fourierOperator.fourierNeuralOperatorLayer import fourierNeuralOperatorLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


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
        if self.operatorType == 'wavelet': return self.initializeWaveletLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag)
        elif self.operatorType == 'fourier': return self.initializeFourierLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag)
        else: raise ValueError(f"The operator type ({self.operatorType}) must be in {'wavelet'}.")

    def initializeWaveletLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag):
        # Unpack the neural operator parameters.
        encodeHighFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeHighFrequencyProtocol', 'highFreq')  # The protocol for encoding the high frequency signals.
        encodeLowFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeLowFrequencyProtocol', 'lowFreq')  # The protocol for encoding the low frequency signals.
        waveletType = neuralOperatorParameters['wavelet'].get('waveletType', None)  # The type of wavelet to use for the wavelet transform.

        # Hardcoded parameters.
        numDecompositions = waveletNeuralOperatorLayer.max_decompositions(signal_length=sequenceLength, wavelet_name=waveletType, minSignalLength=8)  # Number of decompositions for the waveletType transform.
        activationMethod = 'none' if reversibleFlag else emotionModelWeights.getIrreversibleActivation()  # The activation method to use.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        learningProtocol = 'rCNN' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.
        mode = 'periodization'  # Mode: 'zero' (lossy), 'symmetric' (lossy), 'reflect' (lossy), or 'periodization' (lossless).

        return waveletNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals, numOutputSignals=self.numOutputSignals, numDecompositions=numDecompositions,
                                          waveletType=waveletType, mode=mode, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                          encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol)

    def initializeFourierLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag):
        # Unpack the neural operator parameters.
        encodeImaginaryFrequencies = neuralOperatorParameters.get('encodeImaginaryFrequencies', True)
        encodeRealFrequencies = neuralOperatorParameters.get('encodeRealFrequencies', True)

        # Hardcoded parameters.
        activationMethod = 'none' if reversibleFlag else emotionModelWeights.getIrreversibleActivation()  # The activation method to use.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        learningProtocol = 'rCNN' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.

        return fourierNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals, numOutputSignals=self.numOutputSignals, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod,
                                          skipConnectionProtocol=skipConnectionProtocol, encodeRealFrequencies=encodeRealFrequencies, encodeImaginaryFrequencies=encodeImaginaryFrequencies, learningProtocol=learningProtocol)
