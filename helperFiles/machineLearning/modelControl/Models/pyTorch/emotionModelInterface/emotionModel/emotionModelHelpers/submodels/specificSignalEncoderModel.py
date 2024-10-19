from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.transformerHelpers.ebbinghausInterpolation import ebbinghausInterpolation


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, numOperatorLayers, numInputSignals, activationMethod, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(sequenceLength=encodedDimension, numInputSignals=numInputSignals, numOutputSignals=numInputSignals, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.learningProtocol = neuralOperatorParameters['wavelet']['learningProtocol']  # The learning protocol for the neural operator.
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
            if self.learningProtocol == 'rCNN': self.initialProcessingLayers.append(self.postProcessingLayerCNN(numSignals=numInputSignals))
            else: self.initialProcessingLayers.append(self.postProcessingLayerFC(numSignals=numInputSignals, sequenceLength=encodedDimension))

            # Create the final layers.
            self.finalNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            if self.learningProtocol == 'rCNN': self.finalProcessingLayers.append(self.postProcessingLayerCNN(numSignals=numInputSignals))
            else: self.finalProcessingLayers.append(self.postProcessingLayerFC(numSignals=numInputSignals, sequenceLength=encodedDimension))

        # The ebbinghaus interpolation model.
        self.ebbinghausInterpolation = ebbinghausInterpolation(numSignals=numInputSignals, encodedDimension=encodedDimension)

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def learnedInterpolation(self, signalData):
        """ signalData: batchSize, numSignals, signalSpecificLength* """
        interpolatedSignalData, missingDataMask = self.ebbinghausInterpolation(signalData)
        # interpolatedSignalData: batchSize, numSignals, encodedDimension

        return interpolatedSignalData, missingDataMask

    def signalSpecificInterface(self, signalData, initialModel):
        if initialModel: return self.initialLearning(signalData, self.initialNeuralLayers, self.initialProcessingLayers)
        else: return self.initialLearning(signalData, self.finalNeuralLayers, self.finalProcessingLayers)

    def initialLearning(self, signalData, neuralLayers, processingLayers):
        # For each initial layer.
        for layerInd in range(self.numOperatorLayers):
            if reversibleInterface.forwardDirection:
                # Apply the neural operator layer with activation.
                signalData = neuralLayers[layerInd](signalData)
                signalData = self.activationFunction(signalData, layerInd % 2 == 0)

                # Apply the post-processing layer.
                signalData = processingLayers[layerInd](signalData)
            else:
                # Apply the post-processing layer.
                pseudoLayerInd = self.numOperatorLayers - layerInd - 1
                signalData = processingLayers[pseudoLayerInd](signalData)

                # Apply the neural operator layer with activation.
                signalData = self.activationFunction(signalData, pseudoLayerInd % 2 == 0)
                signalData = neuralLayers[pseudoLayerInd](signalData)

        return signalData.contiguous()
