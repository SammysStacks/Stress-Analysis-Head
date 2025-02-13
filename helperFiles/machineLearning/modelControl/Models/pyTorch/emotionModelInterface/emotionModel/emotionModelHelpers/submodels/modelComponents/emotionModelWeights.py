import math

import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolutionLayer import reversibleConvolutionLayer


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
    def getInitialPhysiologicalProfile(numExperiments):
        # Initialize the health profile.
        healthProfile = torch.randn(numExperiments, modelConstants.profileDimension, dtype=torch.float64)
        emotionModelWeights.healthInitialization(healthProfile)
        healthProfile = nn.Parameter(healthProfile)

        return healthProfile

    @staticmethod
    def healthInitialization(healthProfile):
        nn.init.uniform_(healthProfile, a=-modelConstants.initialProfileAmp, b=modelConstants.initialProfileAmp)

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
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength, numLayers=1):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    def postProcessingLayerCNN(self, numSignals):
        return self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False)

    @staticmethod
    def postProcessingLayerFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="SoftSign", addBias=False)

    def healthGeneration(self, numOutputFeatures):
        if numOutputFeatures < modelConstants.profileDimension: raise ValueError(f"Number of outputs ({numOutputFeatures}) must be greater than inputs ({modelConstants.profileDimension})")
        numUpSamples = int(math.log2(numOutputFeatures // modelConstants.profileDimension))

        layers = []
        for i in range(numUpSamples): layers.append(self.linearModel(numInputFeatures=modelConstants.profileDimension*2**i, numOutputFeatures=modelConstants.profileDimension*2**(i+1), activationMethod='SoftSign', addBias=False, addResidualConnection=False))

        layers.extend([
            self.linearModel(numInputFeatures=numOutputFeatures, numOutputFeatures=numOutputFeatures, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.linearModel(numInputFeatures=numOutputFeatures, numOutputFeatures=numOutputFeatures, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.linearModel(numInputFeatures=numOutputFeatures, numOutputFeatures=numOutputFeatures, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=4, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False),
        ])

        # Construct the profile generation model.
        return nn.Sequential(*layers)

    # ------------------- Emotion/Activity Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerFC___(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    def postProcessingLayerCNN___(self, numSignals):
        return self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=numSignals, strides=1, convType='conv1D', activationMethod="boundedExp", numLayers=None, addBias=False)

    def skipConnectionCNN(self, numSignals):
        return self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=numSignals, strides=1, convType='conv1D', activationMethod="boundedExp", numLayers=None, addBias=False)

    @staticmethod
    def skipConnectionFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def getSubjectSpecificBasicEmotionWeights(numBasicEmotions, numSubjects):
        basicEmotionWeights = torch.randn(numSubjects, numBasicEmotions)
        basicEmotionWeights = basicEmotionWeights / basicEmotionWeights.sum(dim=-1, keepdim=True)

        return nn.Parameter(basicEmotionWeights)

    # ------------------- Universal Architectures ------------------- #

    @staticmethod
    def getReversibleActivation(): return 'reversibleLinearSoftSign'  # reversibleLinearSoftSign

    @staticmethod
    def getIrreversibleActivation(): return 'boundedExp'
