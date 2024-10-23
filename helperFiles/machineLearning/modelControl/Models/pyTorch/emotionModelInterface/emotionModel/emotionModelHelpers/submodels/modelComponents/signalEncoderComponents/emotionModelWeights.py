import torch
from matplotlib import pyplot as plt
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.convolutionalHelpers import convolutionalHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolution import reversibleConvolution
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLinearLayer import reversibleLinearLayer


class emotionModelWeights(convolutionalHelpers):

    def __init__(self):
        super(emotionModelWeights, self).__init__()

    def linearModel(self, numInputFeatures=1, numOutputFeatures=1, activationMethod='none', layerType='fc', addBias=False):
        linearLayer = self.weightInitialization.initialize_weights(nn.Linear(numInputFeatures, numOutputFeatures, bias=addBias), activationMethod='none', layerType=layerType)
        if activationMethod == 'none': return linearLayer

        return nn.Sequential(linearLayer, activationFunctions.getActivationMethod(activationMethod))

    # ------------------- Ebbinghaus Forgetting Curve ------------------- #

    @staticmethod
    def ebbinghausDecayPoly(deltaTimes, signalWeights):
        return signalWeights.pow(2)/(1 + deltaTimes.pow(2))

    @staticmethod
    def ebbinghausDecayExp(deltaTimes, signalWeights):
        return signalWeights.pow(2)*torch.exp(-deltaTimes.pow(2))

    # ------------------- Physiological Profile ------------------- #

    def getInitialPhysiologicalProfile(self, numExperiments, encodedDimension):
        # Initialize the physiological profile in the frequency domain.
        imaginaryFourierData = torch.randn(numExperiments, encodedDimension//2 + 1, dtype=torch.float64)/2
        realFourierData = torch.randn(numExperiments, encodedDimension//2 + 1, dtype=torch.float64)/2
        fourierData = realFourierData + 1j * imaginaryFourierData

        # Reconstruct the spatial data.
        physiologicalProfile = torch.fft.irfft(fourierData, n=encodedDimension, dim=-1, norm='ortho')
        physiologicalProfile = self.smoothingFilter(physiologicalProfile, kernelSize=7)

        return nn.Parameter(physiologicalProfile)

    @staticmethod
    def smoothingFilter(data, kernelSize):
        # Add batch and channel dimensions for conv1d
        assert len(data.size()) == 2, "The data must have two dimensions: batch, sequenceDimension."
        kernel = torch.ones((1, 1, kernelSize), dtype=torch.float64) / kernelSize
        data = data.unsqueeze(1)

        # Apply the convolution
        filtered_data = torch.nn.functional.conv1d(data, kernel, padding=kernel.size(-1) // 2)

        # Remove batch and channel dimensions
        return filtered_data.squeeze()

    # ------------------- Wavelet Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=3, numLayers=1, activationMethod=emotionModelWeights.getActivationType())

    def neuralWeightFCC(self, inChannel=1, outChannel=2, finalFrequencyDim=46):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((outChannel, inChannel, finalFrequencyDim)))
        self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel * finalFrequencyDim, fan_out=outChannel * finalFrequencyDim)
        assert False, "The neuralWeightFCC method is not yet implemented."

    @staticmethod
    def reversibleNeuralWeightCNN(inChannel=1):
        return reversibleConvolution(numChannels=inChannel, kernelSize=3, activationMethod=emotionModelWeights.getActivationType(), numLayers=1)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Signal Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerCNN(numSignals=1):
        return reversibleConvolution(numChannels=numSignals, kernelSize=5, activationMethod=emotionModelWeights.getActivationType(), numLayers=1)

    @staticmethod
    def postProcessingLayerFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=5, numLayers=1, activationMethod=emotionModelWeights.getActivationType())

    @staticmethod
    def getActivationType(): return 'nonLinearAddition'


if __name__ == "__main__":
    # General parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 3, 256

    # Initialize the model weights.
    modelWeights = emotionModelWeights()
    modelWeights.getInitialPhysiologicalProfile(numExperiments=_batchSize, encodedDimension=_sequenceLength)
