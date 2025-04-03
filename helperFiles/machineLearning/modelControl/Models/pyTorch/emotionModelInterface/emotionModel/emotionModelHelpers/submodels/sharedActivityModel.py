import copy

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class sharedActivityModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, numLayers, neuralOperatorParameters):
        super(sharedActivityModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=1, numOutputSignals=1, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numLayers = numLayers  # The number of model layers.

        numIgnoredSharedHF = self.neuralOperatorParameters['wavelet']['numIgnoredSharedHF']
        # Only apply a transformation to the lowest of the high frequency decompositions.
        self.neuralOperatorParameters['wavelet']['encodeHighFrequencyProtocol'] = f'highFreq-{0}-{numIgnoredSharedHF}'

        # The neural layers for the signal encoder.
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True)

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def learningInterface(self, signalData):
        # Apply the neural operator layer with activation.
        return self.neuralLayers(signalData).contiguous()
