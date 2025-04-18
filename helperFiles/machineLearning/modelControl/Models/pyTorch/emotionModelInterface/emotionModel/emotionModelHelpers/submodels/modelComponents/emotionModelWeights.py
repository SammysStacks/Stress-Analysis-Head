import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class emotionModelWeights(convolutionalHelpers):

    def __init__(self):
        super(emotionModelWeights, self).__init__()

    @staticmethod
    def linearModel(numInputFeatures=1, numOutputFeatures=1, activationMethod='none', addBias=False, addResidualConnection=False):
        linearLayer = nn.Linear(numInputFeatures, numOutputFeatures, bias=addBias)
        if activationMethod == 'none': return linearLayer

        linearLayer = nn.Sequential(linearLayer, activationFunctions.getActivationMethod(activationMethod))
        if addResidualConnection: return ResNet(module=linearLayer)
        return linearLayer

    # ------------------- Health Profile ------------------- #

    @staticmethod
    def getInitialPhysiologicalProfile(numExperiments, dimension):
        # Initialize the health profile.
        healthProfile = torch.randn(numExperiments, dimension, dtype=torch.float64)
        emotionModelWeights.healthInitialization(healthProfile)
        healthProfile = nn.Parameter(healthProfile)

        return healthProfile

    @staticmethod
    def healthInitialization(healthProfile):
        nn.init.uniform_(healthProfile, a=-modelConstants.userInputParams['initialProfileAmp'], b=modelConstants.userInputParams['initialProfileAmp'])

    # ------------------- Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Signal Encoding Architectures ------------------- #

    @staticmethod
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength, numLayers):
        return reversibleLieLayer(numSignals=numSignals, sequenceLength=sequenceLength, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    def healthGeneration(self):
        layers = []
        for _ in range(4):
            layers.append(self.convolutionalFilters_resNetBlocks(numResNets=3, numBlocks=3, numChannels=[1, 1], kernel_sizes=3, dilations=[1, 1, 1], groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
            layers.append(self.convolutionalFilters_resNetBlocks(numResNets=3, numBlocks=3, numChannels=[1, 1], kernel_sizes=3, dilations=[1, 2, 1], groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))

        # Construct the profile generation model.
        return nn.Sequential(*layers)

    def fourierAdjustments(self):
        layers = []
        for _ in range(4):
            layers.append(self.convolutionalFilters_resNetBlocks(numResNets=3, numBlocks=3, numChannels=[2, 2], kernel_sizes=3, dilations=[1, 1, 1], groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
            layers.append(self.convolutionalFilters_resNetBlocks(numResNets=3, numBlocks=3, numChannels=[2, 2], kernel_sizes=3, dilations=[1, 2, 1], groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))

        # Construct the profile generation model.
        return nn.Sequential(*layers)

    # ------------------- Emotion/Activity Encoding Architectures ------------------- #

    def skipConnectionCNN(self, numSignals):
        return self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=numSignals, strides=1, convType='conv1D', activationMethod="boundedExp", numLayers=None, addBias=False)

    @staticmethod
    def skipConnectionFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def getEmotionSpecificBasicEmotionWeights(numEmotions, numBasicEmotions):
        basicEmotionWeights = torch.zeros(1, numEmotions, numBasicEmotions, 1)
        return nn.Parameter(basicEmotionWeights)

    @staticmethod
    def getSubjectSpecificBasicEmotionWeights(numSubjects, numBasicEmotions):
        subjectSpecificWeights = torch.zeros(numSubjects, 1, numBasicEmotions, 1)
        return nn.Parameter(subjectSpecificWeights)

    # ------------------- Universal Architectures ------------------- #

    @staticmethod
    def getReversibleActivation(): return 'reversibleLinearSoftSign'  # reversibleLinearSoftSign

    @staticmethod
    def getIrreversibleActivation(): return 'none'
