import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolutionLayer import reversibleConvolutionLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLinearLayer import reversibleLinearLayer


class emotionModelWeights(convolutionalHelpers):

    def __init__(self):
        super(emotionModelWeights, self).__init__()

    @staticmethod
    def linearModel(numInputFeatures=1, numOutputFeatures=1, activationMethod='none', addBias=False):
        linearLayer = nn.Linear(numInputFeatures, numOutputFeatures, bias=addBias)
        if activationMethod == 'none': return linearLayer

        return nn.Sequential(linearLayer, activationFunctions.getActivationMethod(activationMethod))

    # ------------------- Ebbinghaus Forgetting Curve ------------------- #

    @staticmethod
    def ebbinghausDecayPoly(deltaTimes, signalWeights):
        return signalWeights.pow(2) / (1 + deltaTimes.pow(2))

    @staticmethod
    def ebbinghausDecayExp(deltaTimes, signalWeights):
        return signalWeights.pow(2) * torch.exp(-deltaTimes.pow(2))

    # ------------------- Physiological Profile ------------------- #

    @staticmethod
    def getInitialPhysiologicalProfile(numExperiments, encodedDimension):
        # Initialize the physiological profile.
        physiologicalProfile = torch.randn(numExperiments, encodedDimension, dtype=torch.float64)
        physiologicalProfile = nn.init.normal_(physiologicalProfile, mean=0, std=0.5)

        # Initialize the physiological profile as a parameter.
        physiologicalProfile = nn.Parameter(physiologicalProfile)
        physiologicalProfile.register_hook(emotionModelWeights.physiologicalHook)

        return physiologicalProfile

    @staticmethod
    def physiologicalHook(grad): return grad*10

    # ------------------- Neural Operator Architectures ------------------- #

    @staticmethod
    def reversibleNeuralWeightRFC(numSignals, sequenceLength, activationMethod):
        activationMethod, switchActivationDirection = activationMethod.split('_')
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength, numLayers=1, activationMethod='none', switchActivationDirection=switchActivationDirection == "True")

    @staticmethod
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength, activationMethod):
        activationMethod, switchActivationDirection = activationMethod.split('_')
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=9, numLayers=1, activationMethod='none', switchActivationDirection=switchActivationDirection == "True")

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Reversible Signal Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength, activationMethod, switchActivationDirection):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=9, numLayers=1, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection)

    @staticmethod
    def postProcessingLayerRFC(numSignals, sequenceLength, activationMethod, switchActivationDirection):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength, numLayers=1, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection)

    # ------------------- Emotion/Activity Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    def postProcessingLayerCNN(self, numSignals):
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
    def getActivationType(): return 'reversibleLinearSoftSign'
