from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, sequenceLength, numOperatorLayers, numInputSignals, activationMethod, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(sequenceLength=sequenceLength, numInputSignals=numInputSignals, numOutputSignals=numInputSignals, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.numOperatorLayers = numOperatorLayers  # The number of operator layers to use.
        self.operatorType = operatorType  # The type of operator to use for the neural operator.

        # The neural layers for the signal encoder.
        self.initialProcessingLayers = nn.ModuleList()
        self.finalProcessingLayers = nn.ModuleList()
        self.initialNeuralLayers = nn.ModuleList()
        self.finalNeuralLayers = nn.ModuleList()

        for layerInd in range(self.numOperatorLayers):
            # Create the initial layers.
            self.initialNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            self.initialProcessingLayers.append(self.postProcessingLayer(inChannel=numInputSignals, groups=numInputSignals))

            # Create the final layers.
            self.finalNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            self.finalProcessingLayers.append(self.postProcessingLayer(inChannel=numInputSignals, groups=numInputSignals))

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def signalSpecificInterface(self, signalData, initialModel):
        if initialModel: return self.initialLearning(signalData, self.initialNeuralLayers, self.initialProcessingLayers)
        else: return self.initialLearning(signalData, self.finalNeuralLayers, self.finalProcessingLayers)

    def initialLearning(self, signalData, neuralLayers, processingLayers):
        # For each initial layer.
        for layerInd in range(self.numOperatorLayers):
            if reversibleInterface.forwardDirection:
                # signalData = neuralLayers[layerInd](signalData)
                # signalData = self.activationFunctions(signalData)

                signalData = processingLayers[layerInd](signalData)
            else:
                signalData = processingLayers[layerInd](signalData)

                # signalData = self.activationFunctions(signalData)
                # signalData = neuralLayers[layerInd](signalData)

        return signalData
