from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class specificEmotionModel(neuralOperatorInterface):

    def __init__(self, numSubjects, numBasicEmotions, encodedDimension, numEmotions, numModelLayers, numSpecificEncoderLayers, operatorType, learningProtocol, neuralOperatorParameters):
        super(specificEmotionModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=1, numInputSignals=numBasicEmotions, numOutputSignals=numBasicEmotions, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions to encode.
        self.numModelLayers = numModelLayers  # The number of model layers to use.
        self.numSpecificEncoderLayers = numSpecificEncoderLayers  # The golden ratio for the model.
        self.numEmotions = numEmotions  # The number of signals to encode.
        self.numSubjects = numSubjects  # The number of subjects to encode.

        # The neural layers for the signal encoder.
        self.spatialLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        for layerInd in range(1 + self.numModelLayers // self.numSpecificEncoderLayers): self.addLayer()

        # Initialize the basic emotion weight.
        self.basicEmotionWeights = self.getSubjectSpecificBasicEmotionWeights(numBasicEmotions=numBasicEmotions, numSubjects=numSubjects)
        
        # Assert the validity of the input parameters.
        assert self.numModelLayers % self.numSpecificEncoderLayers == 0, "The number of model layers must be divisible by the golden ratio."
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

        # Initialize loss holders.
        self.trainingLosses_emotionPrediction = None
        self.testingLosses_emotionPrediction = None
        self.resetModel()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def resetModel(self):
        # Emotion loss holders.
        self.trainingLosses_emotionPrediction = []  # List of list of prediction training losses. Dim: numEpochs
        self.testingLosses_emotionPrediction = []  # List of list of prediction testing losses. Dim: numEpochs

    def addLayer(self):
        # Create the layers.
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=False))
        if self.learningProtocol == 'FC': self.spatialLayers.append(self.postSpatialLayerFC(sequenceLength=self.encodedDimension))
        elif self.learningProtocol == 'CNN': self.spatialLayers.append(self.postSpatialLayerCNN(numSignals=self.numSignals))
        else: raise "The learning protocol is not yet implemented."

    def calculateEmotionProfile(self, basicEmotionProfile, subjectInds):
        batchSize, numEmotions, numBasicEmotions, encodedDimension = basicEmotionProfile.size()

        # Calculate the subject-specific weights.
        subjectSpecificWeights = self.basicEmotionWeights[subjectInds]  # batchSize, numBasicEmotions
        subjectSpecificWeights = subjectSpecificWeights / subjectSpecificWeights.sum(dim=-1, keepdim=True)
        subjectSpecificWeights = subjectSpecificWeights.view(batchSize, 1, numBasicEmotions, 1)
        # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension
        # subjectSpecificWeights: batchSize, 1, numBasicEmotions, 1
        # basicEmotionWeights: numSubjects, numBasicEmotions
        # subjectInds: batchSize

        # Calculate the emotion profile.
        emotionProfile = (basicEmotionProfile * subjectSpecificWeights).sum(dim=2)
        # emotionProfile: batchSize, numEmotions, encodedDimension

        return emotionProfile

    def learningInterface(self, layerInd, signalData, compilingFunction):
        # Apply the neural operator layer with activation.
        signalData = self.neuralLayers[layerInd](signalData)
        signalData = self.spatialLayers[layerInd](signalData)

        return signalData
