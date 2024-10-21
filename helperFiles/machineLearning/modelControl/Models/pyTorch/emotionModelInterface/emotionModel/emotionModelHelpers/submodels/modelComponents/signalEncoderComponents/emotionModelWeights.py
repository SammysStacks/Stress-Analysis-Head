import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers
from ..reversibleComponents.reversibleConvolution import reversibleConvolution
from ..reversibleComponents.reversibleLinearLayer import reversibleLinearLayer


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

    def timeDependantSignalWeights(self, numSignals):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn(1, numSignals, 1, 1))
        return self.weightInitialization.xavierNormalInit(parameter, fan_in=1, fan_out=1)

    # ------------------- Wavelet Neural Operator Architectures ------------------- #

    @staticmethod
    def neuralWeightFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=13, numLayers=4, activationMethod=emotionModelWeights.getActivationType())

    def neuralWeightFCC(self, inChannel=1, outChannel=2, finalFrequencyDim=46):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((outChannel, inChannel, finalFrequencyDim)))
        parameter = self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel * finalFrequencyDim, fan_out=outChannel * finalFrequencyDim)
        assert False, "The neuralWeightFCC method is not yet implemented."

    @staticmethod
    def reversibleNeuralWeightCNN(inChannel=1):
        return reversibleConvolution(numChannels=inChannel, kernelSize=13, activationMethod=emotionModelWeights.getActivationType(), numLayers=4)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    # ------------------- Signal Encoding Architectures ------------------- #

    @staticmethod
    def postProcessingLayerCNN(numSignals=1):
        return reversibleConvolution(numChannels=numSignals, kernelSize=13, activationMethod=emotionModelWeights.getActivationType(), numLayers=4)

    @staticmethod
    def postProcessingLayerFC(numSignals, sequenceLength):
        return reversibleLinearLayer(numSignals=numSignals, sequenceLength=sequenceLength, kernelSize=13, numLayers=4, activationMethod=emotionModelWeights.getActivationType())

    @staticmethod
    def getActivationType(): return 'nonLinearAddition'
