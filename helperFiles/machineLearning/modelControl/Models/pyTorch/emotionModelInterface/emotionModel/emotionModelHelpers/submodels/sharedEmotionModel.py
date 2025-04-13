import copy

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class sharedEmotionModel(neuralOperatorInterface):

    def __init__(self, operatorType, numBasicEmotions, encodedDimension, numLayers, neuralOperatorParameters):
        super(sharedEmotionModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=numBasicEmotions, numOutputSignals=numBasicEmotions, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions.
        self.numLayers = numLayers  # The number of model layers.

        numIgnoredSharedHF = modelConstants.userInputParams['numIgnoredSharedHF']
        # Only apply a transformation to the lowest of the high frequency decompositions.
        self.neuralOperatorParameters['wavelet']['encodeHighFrequencyProtocol'] = f'highFreq-{0}-{numIgnoredSharedHF}'

        # The neural layers for the signal encoder.
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters)

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def learningInterface(self, signalData):
        # Apply the neural operator layer with activation.
        return self.neuralLayers(signalData).contiguous()
