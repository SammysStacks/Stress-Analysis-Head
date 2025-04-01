import numpy as np
import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleLieLayerInterface(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, numLayers, activationMethod):
        super(reversibleLieLayerInterface, self).__init__()
        # General parameters.
        self.numParams = int(sequenceLength * (sequenceLength - 1) / 2)  # The number of free parameters in the model.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.optimalForwardFirst = False  # Whether to apply the forward pass first.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T

        # Initialize the neural layers.
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()
        self.activationFunction = nn.ModuleList()

    def getGivensAngles(self, layerInd):
        return torch.pi * torch.tanh(self.givensRotationParams[layerInd]) / 2  # [-pi/2, pi/2], Dim: numSignals, numParams

    @staticmethod
    def getInverseAngleParams(givensAngles):
        return torch.atanh(2 * torch.as_tensor(givensAngles) / torch.pi)

    def getJacobianScalar(self):
        return 1.0 + 0.05 * torch.tanh(self.jacobianParameter)

    # ------------------- Helper Methods ------------------- #

    @staticmethod
    def initializeJacobianParams(numSignals):
        return nn.Parameter(torch.zeros((1, numSignals, 1)))

    def applyManifoldScale(self, inputData):
        scalarValues = self.getJacobianScalar().expand_as(inputData)
        if reversibleInterface.forwardDirection: return inputData * scalarValues
        else: return inputData / scalarValues

    def getLinearParams(self, layerInd):
        givensAngles = self.getGivensAngles(layerInd)  # Dim: numSignals, numParams
        normalizationFactors = self.getJacobianScalar().flatten()  # Dim: numSignals

        return givensAngles, normalizationFactors

    # ------------------------------------------------------------ #

    def getAllLinearParams(self):
        allGivensAngles, allScaleFactors = [], []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles, normalizationFactors = self.getLinearParams(layerInd)
                allScaleFactors.append(normalizationFactors.unsqueeze(-1).detach().cpu().numpy().astype(np.float16))
                allGivensAngles.append(givensAngles.detach().cpu().numpy().astype(np.float16))
            allGivensAngles = np.asarray(allGivensAngles)
            allScaleFactors = np.asarray(allScaleFactors)

        return allGivensAngles, allScaleFactors

    def getNumFreeParams(self):
        allNumFreeParams = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                angularMask = 0 != self.givensRotationParams[layerInd]
                numSignalParameters = angularMask.sum(dim=-1, keepdim=True)  # Dim: numSignals, 1
                allNumFreeParams.append(numSignalParameters.detach().cpu().numpy().astype(np.int32))
                # allNumFreeParams: numLayers, numSignals, numFreeParams=1

        return allNumFreeParams

    def getFeatureParams(self):
        givensAnglesFeatureNames = self.getFeatureNames()
        allGivensAnglesFeatures = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles, normalizationFactors = self.getLinearParams(layerInd)  # Dim: numSignals, numParams
                normalizationFactors = normalizationFactors.reshape(self.numSignals, 1)  # Dim: numSignals, numParams=1
                givensAngles = givensAngles * 180 / torch.pi  # Convert to degrees

                # Calculate the mean, variance, and range of the Givens angles.
                givensAnglesRange = givensAngles.max(dim=-1).values - givensAngles.min(dim=-1).values  # Dim: numSignals
                givensAnglesVar = givensAngles.var(dim=-1).cpu().detach().numpy().astype(np.float16)  # Dim: numSignals
                givensAnglesRange = givensAnglesRange.cpu().detach().numpy().astype(np.float16)

                # Calculate the mean, variance, and range of the scaling factors.
                normalizationFactorsVar = normalizationFactors.var(dim=0).cpu().detach().numpy().astype(np.float16)  # Dim: numSignals

                # Combine the features. Return dimension: numFeatures, numValues
                allGivensAnglesFeatures.append([givensAnglesVar, givensAnglesRange, normalizationFactorsVar])

        return givensAnglesFeatureNames, allGivensAnglesFeatures

    @staticmethod
    def getFeatureNames():
        return ["Angular variance", "Angular range", "Normalization factor variance"]

    def getAllActivationParams(self):
        with torch.no_grad():
            allActivationParams = []
            for layerInd in range(self.numLayers):
                infiniteBound, linearity, convergentPoint = self.activationFunction[layerInd].getActivationParams()
                allActivationParams.append([infiniteBound.detach().cpu().item(), linearity.detach().cpu().item(), convergentPoint.detach().cpu().item()])
            allActivationParams = np.asarray(allActivationParams)

        return allActivationParams

    def geAllActivationCurves(self, x_min=-1.5, x_max=1.5, num_points=100):
        xs, ys = [], []
        with torch.no_grad():
            for layerInd in range(self.numLayers):
                x, y = self.activationFunction[layerInd].getActivationCurve(x_min, x_max, num_points)
                xs.append(x); ys.append(y)

        return xs, ys

    # ------------------------------------------------------------ #
