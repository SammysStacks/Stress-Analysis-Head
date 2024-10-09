from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, neuralOperatorParameters, sequenceLength, numInitialLayers, numFinalLayers, finalSignalDim, numInputSignals, activationMethod, addBiasTerm):
        super(specificSignalEncoderModel, self).__init__(sequenceLength=sequenceLength, numInputSignals=numInputSignals, numOutputSignals=numInputSignals,
                                                         activationMethod=activationMethod, independentChannels=True, addBiasTerm=addBiasTerm)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.numInitialLayers = numInitialLayers  # The number of initial layers for the signal encoder.
        self.numFinalLayers = numFinalLayers  # The number of final layers for the signal encoder.
        self.finalSignalDim = finalSignalDim  # The final dimension of the signals.

        # The neural layers for the signal encoder.
        self.initialProcessingLayers = nn.ModuleList()
        self.finalProcessingLayers = nn.ModuleList()
        self.initialNeuralLayers = nn.ModuleList()
        self.finalNeuralLayers = nn.ModuleList()

        # Create the initial layers.
        for layerInd in range(self.numInitialLayers):
            self.initialNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            self.initialProcessingLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))

        # Create the final layers.
        for layerInd in range(self.numFinalLayers):
            self.finalNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def initialLearning(self, signalData):
        pass

    def finalLearning(self, signalData):
        pass
