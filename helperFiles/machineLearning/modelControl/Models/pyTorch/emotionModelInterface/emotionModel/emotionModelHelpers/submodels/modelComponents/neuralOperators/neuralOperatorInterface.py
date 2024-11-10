from torch import nn

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

    def getNeuralOperatorLayer(self, neuralOperatorParameters, reversibleFlag, switchActivationDirection):
        # Decide on the neural operator layer.
        if self.operatorType == 'wavelet': return self.initializeWaveletLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=1, useCompiledOperators=True)
        elif self.operatorType == 'fourier': return self.initializeFourierLayer(self.sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=1, useCompiledOperators=True)
        else: raise ValueError(f"The operator type ({self.operatorType}) must be in ['wavelet'].")

    def initializeWaveletLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple, useCompiledOperators=False):
        # Unpack the neural operator parameters.
        encodeHighFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeHighFrequencyProtocol', 'highFreq')  # The protocol for encoding the high frequency signals.
        extraOperators = neuralOperatorParameters['wavelet'].get('extraOperators', ()) if useCompiledOperators else ()  # The extra operators to apply to the wavelet transform.
        encodeLowFrequencyProtocol = neuralOperatorParameters['wavelet'].get('encodeLowFrequencyProtocol', 'lowFreq')  # The protocol for encoding the low frequency signals.
        waveletType = neuralOperatorParameters['wavelet'].get('waveletType', 'bior3.7')  # The type of wavelet to use for the wavelet transform.
        finalSequenceLength = sequenceLength//2

        # Hardcoded parameters.
        activationMethod = f"{emotionModelWeights.getActivationType()}_{switchActivationDirection}"
        mode = 'periodization'  # Mode for the waveletType transform.

        # Hardcoded parameters.
        learningProtocol = 'rCNN' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        numDecompositions = waveletNeuralOperatorLayer.max_decompositions(signal_length=sequenceLength, wavelet_name=waveletType) if learningProtocol not in ['drCNN', 'drFC'] else 1  # Number of decompositions for the waveletType transform.

        # Compile the extra operators.
        compiledOperators = self.compileExtraOperators(finalSequenceLength, extraOperators, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=2)

        return waveletNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals*layerMultiple, numOutputSignals=self.numOutputSignals*layerMultiple, numDecompositions=numDecompositions,
                                          waveletType=waveletType, mode=mode, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                          encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol, extraOperators=compiledOperators)

    def initializeFourierLayer(self, sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple, useCompiledOperators=False):
        # Unpack the neural operator parameters.
        extraOperators = neuralOperatorParameters.get('extraOperators', ()) if useCompiledOperators else ()
        encodeImaginaryFrequencies = neuralOperatorParameters.get('encodeImaginaryFrequencies', True)
        encodeRealFrequencies = neuralOperatorParameters.get('encodeRealFrequencies', True)
        finalSequenceLength = sequenceLength//2 + 1

        # Compile the extra operators.
        compiledOperators = self.compileExtraOperators(finalSequenceLength, extraOperators, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=2)

        # Hardcoded parameters.
        learningProtocol = 'rFC' if reversibleFlag else 'FC'  # The protocol for learning the wavelet data.
        skipConnectionProtocol = 'none' if reversibleFlag else 'CNN'  # The protocol for the skip connections.
        activationMethod = f"{emotionModelWeights.getActivationType()}_{switchActivationDirection}"  # The activation method to use.

        return fourierNeuralOperatorLayer(sequenceLength=sequenceLength, numInputSignals=self.numInputSignals*layerMultiple, numOutputSignals=self.numOutputSignals*layerMultiple, addBiasTerm=self.addBiasTerm, activationMethod=activationMethod,
                                          skipConnectionProtocol=skipConnectionProtocol, encodeRealFrequencies=encodeRealFrequencies, encodeImaginaryFrequencies=encodeImaginaryFrequencies, learningProtocol=learningProtocol, extraOperators=compiledOperators)

    def compileExtraOperators(self, sequenceLength, extraOperators, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple):
        # Initialize the compiled operators.
        compiledOperators = nn.ModuleList()

        # Compile the extra operators.
        for operator in extraOperators:
            switchActivationDirection = not switchActivationDirection
            if operator == 'wavelet': compiledOperators.append(self.initializeWaveletLayer(sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=layerMultiple, useCompiledOperators=False))
            elif operator == 'fourier': compiledOperators.append(self.initializeFourierLayer(sequenceLength, neuralOperatorParameters, reversibleFlag, switchActivationDirection, layerMultiple=layerMultiple, useCompiledOperators=False))
            else: raise ValueError(f"The extra operator ({operator}) must be in ['wavelet', 'fourier'].")

        return compiledOperators
