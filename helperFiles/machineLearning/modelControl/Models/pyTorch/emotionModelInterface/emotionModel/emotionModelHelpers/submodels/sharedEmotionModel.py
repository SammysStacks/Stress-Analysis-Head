from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class sharedEmotionModel(neuralOperatorInterface):

    def __init__(self, numBasicEmotions, encodedDimension, numModelLayers, operatorType, learningProtocol, neuralOperatorParameters):
        super(sharedEmotionModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numInputSignals=numBasicEmotions, numOutputSignals=numBasicEmotions, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions to encode.
        self.numModelLayers = numModelLayers  # The number of model layers to use.

        # The neural layers for the signal encoder.
        self.processingLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        for layerInd in range(self.numModelLayers): self.addLayer()

        # Assert the validity of the input parameters.
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=False, switchActivationDirection=False))
        if self.learningProtocol == 'CNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=self.numBasicEmotions))
        elif self.learningProtocol == 'FC': self.processingLayers.append(self.postProcessingLayerFC(sequenceLength=self.encodedDimension))
        else: raise "The learning protocol is not yet implemented."

    def learningInterface(self, layerInd, signalData):
        # Reshape the signal data.
        batchSize, numEmotions, signalLength = signalData.shape
        signalData = signalData.view(batchSize*numEmotions, 1, signalLength)

        # Apply the neural operator layer with activation.
        signalData = self.neuralLayers[layerInd](signalData)
        signalData = self.processingLayers[layerInd](signalData)

        # Reshape the signal data.
        signalData = signalData.view(batchSize, numEmotions, signalLength)

        return signalData
