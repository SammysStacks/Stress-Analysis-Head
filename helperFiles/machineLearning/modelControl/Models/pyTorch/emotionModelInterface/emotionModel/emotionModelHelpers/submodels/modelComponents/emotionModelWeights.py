import math

import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolutionLayer import reversibleConvolutionLayer


class emotionModelWeights(convolutionalHelpers):

    def __init__(self):
        super(emotionModelWeights, self).__init__()

    @staticmethod
    def linearModel(numInputFeatures=1, numOutputFeatures=1, activationMethod='none', addBias=False):
        linearLayer = nn.Linear(numInputFeatures, numOutputFeatures, bias=addBias)
        if activationMethod == 'none': return linearLayer

        return nn.Sequential(linearLayer, activationFunctions.getActivationMethod(activationMethod))

    # ------------------- Physiological Profile ------------------- #

    @staticmethod
    def getInitialPhysiologicalProfile(numExperiments):
        # Initialize the physiological profile.
        physiologicalProfile = torch.randn(numExperiments, modelConstants.numEncodedWeights, dtype=torch.float64)
        emotionModelWeights.physiologicalInitialization(physiologicalProfile)
        physiologicalProfile = nn.Parameter(physiologicalProfile)

        return physiologicalProfile

    @staticmethod
    def physiologicalInitialization(physiologicalProfile):
        nn.init.xavier_normal_(physiologicalProfile.data)  # TODO:
        # nn.init.normal_(physiologicalProfile.data, mean=0, std=1/5)

    # ------------------- Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Signal Encoding Architectures ------------------- #

    @staticmethod
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength, numLayers=1):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength*2 - 1, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength, numLayers=1):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength*2 - 1, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    def postProcessingLayerCNN(self, numSignals):
        return self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False)

    @staticmethod
    def postProcessingLayerFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="SoftSign", addBias=False)

    def physiologicalGeneration(self, numOutputFeatures):
        numUpSamples = int(math.log2(numOutputFeatures // modelConstants.numEncodedWeights))
        layers = []

        # Construct the profile generation model.
        layers = [self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False)]
        # layers.append(self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False))
        for i in range(numUpSamples): layers.append(self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=1, numChannels=[1, 2], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
        layers.append(self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=4, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
        return nn.Sequential(*layers)

    @staticmethod
    def gradientHook(grad): return grad

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
        basicEmotionWeights = torch.randn(numSubjects, numBasicEmotions, dtype=torch.float64)
        basicEmotionWeights = basicEmotionWeights / basicEmotionWeights.sum(dim=-1, keepdim=True)

        return nn.Parameter(basicEmotionWeights)

    # ------------------- Universal Architectures ------------------- #

    @staticmethod
    def getReversibleActivation(): return 'reversibleLinearSoftSign'

    @staticmethod
    def getIrreversibleActivation(): return 'boundedExp'
