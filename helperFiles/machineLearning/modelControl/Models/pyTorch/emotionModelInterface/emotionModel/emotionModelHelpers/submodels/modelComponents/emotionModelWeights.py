import torch
from torch import nn

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
    def getInitialPhysiologicalProfile(numExperiments, encodedDimension):
        # Initialize the physiological profile.
        physiologicalProfile = torch.randn(numExperiments, encodedDimension, dtype=torch.float64)
        physiologicalProfile = nn.init.kaiming_normal_(physiologicalProfile)

        # Initialize the physiological profile as a parameter.
        physiologicalProfile = nn.Parameter(physiologicalProfile)

        return physiologicalProfile

    @staticmethod
    def smoothingFilter(data, kernel=(), kernelSize=None):
        # Validate input parameters
        assert len(kernel) != 0 or kernelSize is not None, "The kernel or kernel size must be specified."
        if kernelSize is not None: assert kernelSize % 2 == 1, "The kernel size must be odd."
        if len(kernel) != 0: assert len(kernel) % 2 == 1, "The kernel size must be odd."

        # Define the kernel
        if kernelSize is not None: kernel = torch.ones((1, 1, kernelSize), dtype=data.dtype, device=data.device) / kernelSize
        else: kernel = torch.as_tensor(kernel, dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(0)
        kernel = kernel / kernel.sum(dim=-1)  # Normalize kernel

        # Expand kernel to match data channels
        kernel = kernel.expand(data.size(1), 1, kernel.size(-1))

        # Apply the 1D convolution along the last dimension with padding
        filtered_data = torch.nn.functional.pad(data, (kernel.size(-1) // 2, kernel.size(-1) // 2), mode='reflect')
        filtered_data = torch.nn.functional.conv1d(filtered_data, kernel, padding=0, groups=data.size(1), stride=1, dilation=1)

        return filtered_data

    # ------------------- Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numInputFeatures=sequenceLength, numOutputFeatures=sequenceLength, activationMethod="boundedExp", addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Signal Encoding Architectures ------------------- #

    @staticmethod
    def reversibleNeuralWeightRCNN(numSignals, sequenceLength, numLayers=2):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength*2 - 1, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    @staticmethod
    def postProcessingLayerRCNN(numSignals, sequenceLength, numLayers=2):
        return reversibleConvolutionLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=sequenceLength*2 - 1, numLayers=numLayers, activationMethod=f"{emotionModelWeights.getReversibleActivation()}")

    def physiologicalSmoothing(self):
        return nn.Sequential(
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=3, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="selu", numLayers=None, addBias=False),
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=3, numChannels=[1, 1], kernel_sizes=5, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="selu", numLayers=None, addBias=False),
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=3, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod="selu", numLayers=None, addBias=False),
        )

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
    def getReversibleActivation(): return 'reversibleLinearSoftSign'

    @staticmethod
    def getIrreversibleActivation(): return 'boundedExp'
