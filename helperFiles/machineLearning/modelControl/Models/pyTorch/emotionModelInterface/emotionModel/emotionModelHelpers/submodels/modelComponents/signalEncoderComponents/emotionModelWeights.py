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

        # Initialize the physiological profile as a parameter.
        physiologicalProfile = nn.Parameter(physiologicalProfile)
        physiologicalProfile.register_hook(emotionModelWeights.scaleGradients)

        return physiologicalProfile

    @staticmethod
    def scaleGradients(grad):
        return grad * 100

    @staticmethod
    def smoothingFilter(data, kernel=(), kernelSize=None):
        assert len(data.size()) == 2, "The data must have two dimensions: batch, sequenceDimension."
        if kernelSize is not None: assert kernelSize % 2 == 1, "The kernel size must be odd."
        if kernel is not None: assert len(kernel) % 2 == 1, "The kernel size must be odd."
        assert kernel is not None or kernelSize is not None, "The kernel or kernel size must be specified."
        if kernel is not None: kernel = torch.tensor(kernel)

        # Add batch and channel dimensions for conv1d
        if kernelSize is not None: kernel = torch.ones((1, 1, kernelSize), dtype=torch.float64) / kernelSize
        if kernel is not None: kernel = kernel.unsqueeze(0).unsqueeze(0) / kernel.sum()
        data = data.unsqueeze(1)

        # Apply the convolution
        filtered_data = torch.nn.functional.conv1d(data, kernel, padding=kernel.size(-1) // 2)

        # Remove batch and channel dimensions
        return filtered_data.squeeze()

    # ------------------- Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightRFC(numSignals, sequenceLength, activationMethod):
        activationMethod, switchActivationDirection = activationMethod.split('_')
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength, numLayers=3, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection == "True")

    @staticmethod
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength, activationMethod):
        activationMethod, switchActivationDirection = activationMethod.split('_')
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=3, numLayers=3, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection == "True")

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod=emotionModelWeights.getActivationType(), addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Reversible Signal Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength, activationMethod, switchActivationDirection):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=3, numLayers=3, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection)

    @staticmethod
    def postProcessingLayerRFC(numSignals, sequenceLength, activationMethod, switchActivationDirection):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength, numLayers=3, activationMethod=activationMethod, switchActivationDirection=switchActivationDirection)

    # ------------------- Emotion/Activity Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod=emotionModelWeights.getActivationType(), addBias=False)

    def postProcessingLayerCNN(self, numSignals):
        return self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=numSignals, strides=1, convType='conv1D', activationMethod=emotionModelWeights.getActivationType(), numLayers=None, addBias=False)

    def skipConnectionCNN(self, numSignals):
        return self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[numSignals, numSignals], kernel_sizes=3, dilations=1, groups=numSignals, strides=1, convType='conv1D', activationMethod=emotionModelWeights.getActivationType(), numLayers=None, addBias=False)

    @staticmethod
    def skipConnectionFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod=emotionModelWeights.getActivationType(), addBias=False)

    @staticmethod
    def getSubjectSpecificBasicEmotionWeights(numBasicEmotions, numSubjects):
        basicEmotionWeights = torch.randn(numSubjects, numBasicEmotions, dtype=torch.float64)
        basicEmotionWeights = basicEmotionWeights / basicEmotionWeights.sum(dim=-1, keepdim=True)

        return nn.Parameter(basicEmotionWeights)

    # ------------------- Universal Architectures ------------------- #

    @staticmethod
    def getActivationType(): return 'nonLinearMultiplication'


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256

    # Initialize the model weights.
    modelWeights = emotionModelWeights()
    modelWeights.getInitialPhysiologicalProfile(numExperiments=_batchSize, encodedDimension=_sequenceLength)
