# PyTorch
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.modelHelpers.abnormalConvolutions import abnormalConvolutions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleConvolution import reversibleConvolution


class convolutionalHelpers(abnormalConvolutions):

    def __init__(self):
        super(convolutionalHelpers, self).__init__()

    # ------------------------ General Architectures ----------------------- #

    @staticmethod
    def encodingInterface_reshapeMethod(signalData, transformation, useCheckpoint=False):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = signalData.size()

        # Reshape the data to process each signal separately.
        signalData = signalData.view(batchSize * numSignals, 1, signalDimension)

        # Apply a CNN network.
        if useCheckpoint:
            signalData = checkpoint(transformation, signalData, use_reentrant=False)
        else:
            signalData = transformation(signalData)

        # Return to the initial dimension of the input.
        signalData = signalData.view(batchSize, numSignals, signalDimension)

        return signalData

    @staticmethod
    def encodingInterface_forEach(signalData, transformation, useCheckpoint=False):
        # Initialize the processed data.
        processedData = torch.zeros_like(signalData, device=signalData.mainDevice)

        # For each input channel.
        for signalInd in range(signalData.size(1)):
            # Apply a CNN network.
            if useCheckpoint:
                processedData[:, signalInd:signalInd + 1, :] = checkpoint(transformation, signalData[:, signalInd:signalInd + 1, :], use_reentrant=False)
            else:
                processedData[:, signalInd:signalInd + 1, :] = transformation(signalData[:, signalInd:signalInd + 1, :])

        return processedData

    # --------------- Standard Convolutional Architectures --------------- #

    def convolutionalFilters_semiResNetBlocks(self, numResNets, numBlocks, numChannels, kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D',
                                              activationMethod='selu', scalingFactor=1, secondMethodType='pointwise', finalDim=None, addBias=True):

        if secondMethodType == 'pointwise':
            secondMethod = self.convolutionalFiltersBlocks(numBlocks=numBlocks, numChannels=numChannels, kernel_sizes=kernel_sizes, dilations=dilations,
                                                           groups=groups, strides=strides, convType=secondMethodType, activationMethod=activationMethod, numLayers=None, addBias=False),
        elif secondMethodType == 'upsample':
            secondMethod = nn.Upsample(size=finalDim, mode='linear', align_corners=True)
        else:
            raise ValueError("Second method must be either 'pointwise' or 'upsample'")

        layers = []
        for i in range(numResNets):
            layers.append(addModules(
                firstModule=nn.Sequential(
                    # Convolution architecture: feature engineering
                    self.convolutionalFiltersBlocks(numBlocks=numBlocks, numChannels=numChannels, kernel_sizes=kernel_sizes, dilations=dilations,
                                                    groups=groups, strides=strides, convType=convType, activationMethod=activationMethod, numLayers=None, addBias=addBias),
                ), secondModule=secondMethod, scalingFactor=scalingFactor
            ))

        return nn.Sequential(*layers)

    def convolutionalFilters_resNetBlocks(self, numResNets, numBlocks, numChannels, kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod='selu', numLayers=None, addBias=True):
        if not isinstance(numChannels, list):
            assert numLayers is not None
        else:
            assert numChannels[0] == numChannels[-1] or numChannels[0] == 1, f"For restNets we need the same first and last channel number: {numChannels}"

        layers = []
        for i in range(numResNets):
            layers.append(ResNet(module=nn.Sequential(
                self.convolutionalFiltersBlocks(numBlocks=numBlocks, numChannels=numChannels, kernel_sizes=kernel_sizes, dilations=dilations, groups=groups, strides=strides,
                                                convType=convType, activationMethod=activationMethod, numLayers=numLayers, addBias=addBias),
            )))

        return nn.Sequential(*layers)

    def convolutionalFiltersBlocks(self, numBlocks, numChannels, kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod='selu', numLayers=None, addBias=True):
        if not isinstance(kernel_sizes, list): kernel_sizes = [kernel_sizes] * numBlocks
        if not isinstance(dilations, list): dilations = [dilations] * numBlocks
        if not isinstance(strides, list): strides = [strides] * numBlocks
        if not isinstance(groups, list): groups = [groups] * numBlocks

        layers = []
        for i in range(numBlocks):
            layers.append(self.convolutionalFilters(numChannels=numChannels, kernel_sizes=kernel_sizes[i], dilations=dilations[i], groups=groups[i], strides=strides[i], convType=convType,
                                                    activationMethod=activationMethod, numLayers=numLayers, addBias=addBias))

        return nn.Sequential(*layers)

    def convolutionalFilters(self, numChannels, kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationMethod='selu', numLayers=None, addBias=True):
        # Assert the integrity of the inputs.
        assert isinstance(numChannels, list) or numLayers is not None, f"If numLayers is not provided, numChannels must be a list: {numChannels} {numLayers}"

        # Pointwise parameters.
        if convType == 'pointwise':
            kernel_sizes = self.convolutionalParamCheck(param=kernel_sizes, defaultValue=1)
            dilations = self.convolutionalParamCheck(param=dilations, defaultValue=1)
            strides = self.convolutionalParamCheck(param=strides, defaultValue=1)
            groups = self.convolutionalParamCheck(param=groups, defaultValue=1)
        # Depthwise parameters.
        elif convType == 'depthwise':
            groups = self.convolutionalParamCheck(param=groups, defaultValue=numChannels[0] if isinstance(numChannels, list) else numChannels)

        # Get the parameters in the correct format.
        if numLayers is None: numLayers = len(numChannels) - 1
        if not isinstance(kernel_sizes, list): kernel_sizes = [kernel_sizes] * numLayers
        if not isinstance(numChannels, list): numChannels = [numChannels] * numLayers
        if not isinstance(dilations, list): dilations = [dilations] * numLayers
        if not isinstance(strides, list): strides = [strides] * numLayers
        if not isinstance(groups, list): groups = [groups] * numLayers
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]

        # Assert the integrity of the convolutional layers.
        assert len(kernel_sizes) == len(dilations) == len(groups) == len(strides), "All convolutional parameters must have the same length"

        layers = []
        # For each layer to add.
        for i in range(len(numChannels) - 1):

            # If adding a standard convolutional layer.
            if convType.split("_")[0] in ['conv1D', 'pointwise', 'depthwise']:
                layer = nn.Conv1d(in_channels=numChannels[i], out_channels=numChannels[i + 1], kernel_size=kernel_sizes[i], stride=strides[i],
                                  padding=paddings[i], dilation=dilations[i], groups=groups[i], padding_mode='reflect', bias=addBias)

            elif convType.split("_")[0] == 'reverseConv1D':
                layer = reversibleConvolution(numChannels=numChannels[i], kernelSize=kernel_sizes[i], activationMethod=activationMethod, numLayers=numLayers)
                assert groups[i] == numChannels[i], "The number of groups must equal the number of channels for reversibility (depthwise)."
                assert len(numChannels) == 2, "Reversible convolutions must have two channels."
                activationMethod = 'none'

            # If adding a transposed convolutional layer.
            elif convType.split("_")[0] == 'transConv1D':
                layer = nn.ConvTranspose1d(in_channels=numChannels[i], out_channels=numChannels[i + 1], kernel_size=kernel_sizes[i], stride=strides[i],
                                           padding=paddings[i], dilation=dilations[i], groups=groups[i], padding_mode='zeros', bias=addBias, output_padding=0)

            # If the convolutional type is not recognized.
            else:
                raise ValueError("Unknown convolutional type")

            # Initialize the weights of the convolutional layer.
            layer = self.weightInitialization.initialize_weights(layer, activationMethod=activationMethod, layerType=convType)
            layers.append(layer)

            # Get the activation method.
            activationFunction = activationFunctions.getActivationMethod(activationMethod)

            # Add the activation layer.
            layers.append(activationFunction)

        return nn.Sequential(*layers)

    @staticmethod
    def convolutionalParamCheck(param, defaultValue):
        if isinstance(param, list):
            assert all([paramVal == defaultValue for paramVal in param]), "Pointwise convolutions must have a kernel size of 1"
        elif param is not None:
            assert param == defaultValue, "Pointwise convolutions must have a kernel size of 1"
        else:
            param = defaultValue

        return param


# -------------------------------------------------------------------------- #
# ---------------------------- Pooling Methods ----------------------------- #

class splitPoolingHead(nn.Module):
    def __init__(self, module, poolingLayers):
        super().__init__()
        # General helpers.
        self.poolingLayers = poolingLayers
        self.module = module

    def forward(self, inputs):
        return self.module(inputs, self.poolingLayers)


# -------------------------------------------------------------------------- #

class addModules(torch.nn.Module):
    def __init__(self, firstModule, secondModule, scalingFactor=1, secondModuleScale=1):
        super().__init__()
        # General helpers.
        self.secondModuleScale = secondModuleScale
        self.scalingFactor = scalingFactor
        self.secondModule = secondModule
        self.firstModule = firstModule

    def forward(self, inputs):
        # Return them outputs of the models added together.
        return (self.firstModule(inputs) + self.secondModuleScale * self.secondModule(inputs)) / self.scalingFactor


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        # General helpers.
        self.module = module

    def forward(self, inputs):
        # Return the residual connection.
        return self.module(inputs) + inputs

    # -------------------------------------------------------------------------- #


class independentModelCNN(torch.nn.Module):
    def __init__(self, module, useCheckpoint=False):
        super().__init__()
        # General helpers.
        self.useCheckpoint = useCheckpoint
        self.module = module

    def forward(self, signalData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = signalData.size()

        # Reshape the data to process each signal separately.
        signalData = signalData.view(batchSize * numSignals, 1, signalDimension)

        # Apply a CNN network.
        if self.useCheckpoint:
            signalData = checkpoint(self.module, signalData, use_reentrant=False)
        else:
            signalData = self.module(signalData)

        # Return to the initial dimension of the input.
        signalData = signalData.view(batchSize, numSignals, signalDimension)

        return signalData


class modelCNN(torch.nn.Module):
    def __init__(self, module, useCheckpoint=False):
        super().__init__()
        # General helpers.
        self.useCheckpoint = useCheckpoint
        self.module = module

    def forward(self, signalData):
        # Apply a CNN network.
        if self.useCheckpoint:
            signalData = checkpoint(self.module, signalData, use_reentrant=False)
        else:
            signalData = self.module(signalData)

        return signalData


# -------------------------------------------------------------------------- #

class printDimensionsModel(nn.Module):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def forward(self, inputs):
        print(self.message, inputs.shape)
        return inputs
