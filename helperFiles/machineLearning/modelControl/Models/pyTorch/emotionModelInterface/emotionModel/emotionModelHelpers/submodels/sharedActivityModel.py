from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class sharedActivityModel(neuralOperatorInterface):

    def __init__(self, encodedDimension, numModelLayers, numActivityChannels, operatorType, activationMethod, learningProtocol, neuralOperatorParameters):
        super(sharedActivityModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numInputSignals=numActivityChannels, numOutputSignals=numActivityChannels, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.numActivityChannels = numActivityChannels  # The number of activity channels to encode.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numModelLayers = numModelLayers  # The number of model layers to use.

        # The neural layers for the signal encoder.
        self.processingLayers, self.neuralLayers, self.addingFlags = nn.ModuleList(), nn.ModuleList(), []
        for layerInd in range(self.numModelLayers): self.addLayer()

        # Assert the validity of the input parameters.
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.addingFlags.append(not self.addingFlags[-1] if len(self.addingFlags) != 0 else True)
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=False))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=self.numActivityChannels))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerFC(sequenceLength=self.encodedDimension))
        else: raise "The learning protocol is not yet implemented."

    def learningInterface(self, layerInd, signalData):
        # Apply the neural operator layer with activation.
        signalData = self.neuralLayers[layerInd](signalData)
        signalData = self.activationFunction(signalData, addingFlag=self.addingFlags[layerInd])

        # Apply the post-processing layer.
        signalData = self.processingLayers[layerInd](signalData)

        return signalData
