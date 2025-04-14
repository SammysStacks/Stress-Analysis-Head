import copy

from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class specificEmotionModel(neuralOperatorInterface):

    def __init__(self, operatorType, numSubjects, numBasicEmotions, encodedDimension, emotionNames, numLayers, neuralOperatorParameters):
        super(specificEmotionModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=len(emotionNames)*numBasicEmotions, numOutputSignals=len(emotionNames)*numBasicEmotions, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions to encode.
        self.numEmotions = len(emotionNames)  # The number of emotions.
        self.emotionNames = emotionNames  # The emotion labels.
        self.numSubjects = numSubjects  # The number of subjects in the dataset.
        self.numLayers = numLayers  # The number of layers in the model.

        # Set the wavelet parameters.
        self.neuralOperatorParameters['wavelet']['minWaveletDim'] = encodedDimension // 2

        # The neural layers for the signal encoder.
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters)
        self.basicEmotionWeights = self.getEmotionSpecificBasicEmotionWeights(numEmotions=self.numEmotions, numBasicEmotions=numBasicEmotions) # 1, numEmotions, numBasicEmotions, 1
        # self.subjectSpecificWeights = self.getSubjectSpecificBasicEmotionWeights(numSubjects=numSubjects, numBasicEmotions=self.numBasicEmotions)  # numSubjects, 1, numBasicEmotions, 1
        self.sigmoid = nn.Sigmoid()

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction, self.testingLosses_signalReconstruction = None, None
        self.givensAnglesFeaturesPath, self.normalizationFactorsPath = None, None
        self.activationParamsPath, self.numFreeParams = None, None
        self.resetModel()

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def resetModel(self):
        # Signal encoder reconstruction holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: loadSubmodelEpochs, numTrainingSignals
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: loadSubmodelEpochs, numTestingSignals
        self.givensAnglesFeaturesPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*, numParams
        self.normalizationFactorsPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*
        self.activationParamsPath = []  # List of activation bounds. Dim: loadSubmodelEpochs, numActivations, numActivationParams
        self.numFreeParams = []  # List of the number of free parameters. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*

    def learningInterface(self, signalData):
        # Apply the neural operator layer with activation.
        return self.neuralLayers(signalData).contiguous()

    def calculateEmotionProfile(self, basicEmotionProfile, subjectInds):
        # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension

        # Calculate the emotion profile.
        basicEmotionWeights = self.sigmoid(self.basicEmotionWeights)  # 1, numEmotions, numBasicEmotions, 1
        emotionProfile = (basicEmotionProfile * basicEmotionWeights).sum(dim=2) / basicEmotionWeights.sum(dim=2)  # batchSize, numEmotions, encodedDimension
        # emotionProfile: batchSize, numEmotions, encodedDimension

        # Calculate the emotion profile.
        # subjectSpecificWeights = self.sigmoid(self.subjectSpecificWeights[subjectInds])  # batchSize, 1, numBasicEmotions, 1
        # emotionProfile = (basicEmotionProfile * subjectSpecificWeights).sum(dim=2) / subjectSpecificWeights.sum(dim=2)  # batchSize, numEmotions, encodedDimension
        # emotionProfile: batchSize, numEmotions, encodedDimension

        return emotionProfile
