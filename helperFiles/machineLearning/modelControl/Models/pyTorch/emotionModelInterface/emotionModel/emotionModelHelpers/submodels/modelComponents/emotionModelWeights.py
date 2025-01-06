import math

import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.abnormalConvolutions import subPixelUpsampling1D
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolutionLayer import reversibleConvolutionLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class emotionModelWeights(convolutionalHelpers):

    def __init__(self):
        super(emotionModelWeights, self).__init__()

    @staticmethod
    def linearModel(numInputFeatures=1, numOutputFeatures=1, activationMethod='none', addBias=False, addResidualConnection=False):
        linearLayer = nn.Linear(numInputFeatures, numOutputFeatures, bias=addBias)
        if activationMethod == 'none': return linearLayer

        linearLayer = nn.Sequential(linearLayer, activationFunctions.getActivationMethod(activationMethod))
        if addResidualConnection:
            if numInputFeatures == numOutputFeatures: return ResNet(module=linearLayer)
            linearLayer.append(subPixelUpsampling1D(upscale_factor=numOutputFeatures // numInputFeatures))
            assert False
        return linearLayer

    # ------------------- Health Profile ------------------- #

    @staticmethod
    def getInitialPhysiologicalProfile(numExperiments):
        # Initialize the health profile.
        healthProfile = torch.randn(numExperiments, modelConstants.numEncodedWeights, dtype=torch.float64)
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
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, numLayers=1, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, numLayers=1, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    def postProcessingLayerCNN(self, numSignals):
        return self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False)

    @staticmethod
    def postProcessingLayerFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="SoftSign", addBias=False)

    @staticmethod
    def initializeJacobianParams(numSignals):
        initialValue = -torch.log(3*torch.ones(1))
        if numSignals == 1: return nn.Parameter(initialValue * torch.ones((1, numSignals)))
        else: return nn.Parameter(initialValue * torch.ones((1, numSignals, 1)))

    def healthGeneration(self, numOutputFeatures):
        if numOutputFeatures < modelConstants.numEncodedWeights: raise ValueError(f"Number of outputs ({numOutputFeatures}) must be greater than inputs ({modelConstants.numEncodedWeights})")
        numUpSamples = int(math.log2(numOutputFeatures // modelConstants.numEncodedWeights))

        layers = [
            # Linear model with residual connection.
            self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
            self.linearModel(numInputFeatures=modelConstants.numEncodedWeights, numOutputFeatures=modelConstants.numEncodedWeights, activationMethod='SoftSign', addBias=False, addResidualConnection=True),
        ]

        # Construct the profile generation model.
        for i in range(numUpSamples): layers.append(self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=1, numChannels=[1, 2, 2], kernel_sizes=[[3, 3]], dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
        layers.append(self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=1, numChannels=[1, 2, 2, 2, 1], kernel_sizes=[[3, 3, 3, 3]], dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="SoftSign", numLayers=None, addBias=False))
        return nn.Sequential(*layers)

    @staticmethod
    def getJacobianScalar(jacobianParameter):
        jacobianMatrix = 1/3 + (4/3) * torch.sigmoid(jacobianParameter)
        return jacobianMatrix

    @staticmethod
    def gradientHook(grad): return grad

    def applyManifoldScale(self, healthProfile, healthProfileJacobians):
        scalarValues = self.getJacobianScalar(healthProfileJacobians).expand_as(healthProfile)
        if not reversibleInterface.forwardDirection: return healthProfile / scalarValues
        else: return healthProfile * scalarValues

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
    def getReversibleActivation(): return 'reversibleLinearSoftSign'  # reversibleLinearSoftSign

    @staticmethod
    def getIrreversibleActivation(): return 'boundedExp'
