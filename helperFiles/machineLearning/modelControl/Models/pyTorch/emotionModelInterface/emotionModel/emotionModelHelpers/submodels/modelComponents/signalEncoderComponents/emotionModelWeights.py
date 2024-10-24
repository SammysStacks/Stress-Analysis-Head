import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolution import reversibleConvolution
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
        # Initialize the physiological profile in the frequency domain.
        imaginaryFourierData = torch.randn(numExperiments, encodedDimension // 2 + 1, dtype=torch.float64) / 2
        realFourierData = torch.randn(numExperiments, encodedDimension // 2 + 1, dtype=torch.float64) / 2
        fourierData = realFourierData + 1j * imaginaryFourierData

        # Reconstruct the spatial data.
        physiologicalProfile = torch.fft.irfft(fourierData, n=encodedDimension, dim=-1, norm='ortho')

        return nn.Parameter(physiologicalProfile)

    @staticmethod
    def smoothingFilter(data, kernelSize):
        assert len(data.size()) == 2, "The data must have two dimensions: batch, sequenceDimension."
        assert kernelSize % 2 == 1, "The kernel size must be odd."

        # Add batch and channel dimensions for conv1d
        kernel = torch.ones((1, 1, kernelSize), dtype=torch.float64) / kernelSize
        data = data.unsqueeze(1)

        # Apply the convolution
        filtered_data = torch.nn.functional.conv1d(data, kernel, padding=kernel.size(-1) // 2)

        # Remove batch and channel dimensions
        return filtered_data.squeeze()

    # ------------------- Wavelet Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightRFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=3, numLayers=1, activationMethod=emotionModelWeights.getActivationType())

    @staticmethod
    def reversibleNeuralWeightRCNN(inChannel=1):
        return reversibleConvolution(numChannels=inChannel, kernelSize=3, activationMethod=emotionModelWeights.getActivationType(), numLayers=1)

    @staticmethod
    def neuralWeightFC(sequenceLength):
        return emotionModelWeights.linearModel(numOutputFeatures=sequenceLength, activationMethod=emotionModelWeights.getActivationType(), addBias=False)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Reversible Signal Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerRCNN(numSignals=1):
        return reversibleConvolution(numChannels=numSignals, kernelSize=7, activationMethod=emotionModelWeights.getActivationType(), numLayers=1)

    @staticmethod
    def postProcessingLayerRFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=7, numLayers=1, activationMethod=emotionModelWeights.getActivationType())

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
    def getActivationType(): return 'nonLinearAddition'


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256

    # Initialize the model weights.
    modelWeights = emotionModelWeights()
    modelWeights.getInitialPhysiologicalProfile(numExperiments=_batchSize, encodedDimension=_sequenceLength)
