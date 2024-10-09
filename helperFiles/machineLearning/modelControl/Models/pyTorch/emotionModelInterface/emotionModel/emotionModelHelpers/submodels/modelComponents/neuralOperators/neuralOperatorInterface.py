from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.signalEncoderModules import signalEncoderModules


class neuralOperatorInterface(signalEncoderModules):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, activationMethod, independentChannels, addBiasTerm):
        super().__init__()
        # General parameters.
        self.independentChannels = independentChannels  # Whether to treat each channel independently.
        self.activationMethod = activationMethod  # The activation method to use for the neural operator.
        self.numOutputSignals = numOutputSignals  # The number of output signals.
        self.numInputSignals = numInputSignals  # The number of input signals.
        self.sequenceLength = sequenceLength  # The length of the input signals.
        self.addBiasTerm = addBiasTerm  # Whether to add a bias term to the neural operator.

    def getNeuralOperatorLayer(self, neuralOperatorParameters):
        # The type of operator to use for the neural operator.
        operatorType = neuralOperatorParameters['operatorType']

        # Decide on the neural operator layer.
        if operatorType == 'wavelet': return self.initializeWaveletLayer(neuralOperatorParameters[operatorType])
        else: raise ValueError(f"The operator type ({operatorType}) must be in ['wavelet'].")

    def initializeWaveletLayer(self, neuralOperatorParameters):
        # Unpack the neural operator parameters.
        encodeHighFrequencyProtocol = neuralOperatorParameters['encodeHighFrequencyProtocol']  # The protocol for encoding the high frequency signals.
        encodeLowFrequencyProtocol = neuralOperatorParameters['encodeLowFrequencyProtocol']  # The protocol for encoding the low frequency signals.
        skipConnectionProtocol = neuralOperatorParameters['skipConnectionProtocol']  # The protocol for the skip connections.
        numDecompositions = neuralOperatorParameters['numDecompositions']  # The number of decompositions for the wavelet transform.
        waveletType = neuralOperatorParameters['waveletType']  # The type of wavelet to use for the wavelet transform.
        mode = neuralOperatorParameters['mode']  # The mode for the wavelet transform.

        # Specify the default parameters.
        if encodeHighFrequencyProtocol is None: encodeHighFrequencyProtocol = 'residual'  # The protocol for encoding the high frequency signals.
        if numDecompositions is None: numDecompositions = min(5, waveletNeuralOperatorLayer.max_decompositions(signal_length=self.sequenceLength, wavelet_name=waveletType))  # Number of decompositions for the waveletType transform.
        if mode is None: mode = 'zero'  # Mode for the waveletType transform.

        return waveletNeuralOperatorLayer(sequenceLength=self.sequenceLength, numInputSignals=self.numInputSignals, numOutputSignals=self.numOutputSignals, numDecompositions=numDecompositions,
                                          waveletType=waveletType, mode=mode, addBiasTerm=self.addBiasTerm, activationMethod=self.activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                          encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, independentChannels=self.independentChannels)
