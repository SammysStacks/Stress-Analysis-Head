import numpy as np
import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
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

        # # Get the sub-rotation indices: [X, Y, Z; Q, R, S]
        # self.xrInds, self.xqInds, self.xsInds = (torch.zeros(self.numParams, dtype=torch.int) for _ in range(3))  # [XQ, XR, XS]
        # self.yrInds, self.yqInds, self.ysInds = (torch.zeros(self.numParams, dtype=torch.int) for _ in range(3))  # [YQ, YR, YS]
        # self.zrInds, self.zqInds, self.zsInds = (torch.zeros(self.numParams, dtype=torch.int) for _ in range(3))  # [ZQ, ZR, ZS]
        # for angularLocationsInd in range(self.numParams):
        #     i, j = self.rowInds[angularLocationsInd], self.colInds[angularLocationsInd]
        #     downwardShift = self.sequenceLength - i - 2
        #     upwardShift = -(downwardShift + 1)
        #
        #     # Boolean location flags.
        #     onRightEdge = j == self.sequenceLength - 1
        #     onLeftEdge = abs(i - j) == 1
        #     topRow = i == 0
        #
        #     # Y terms.
        #     yrInd = angularLocationsInd
        #     self.yqInds[angularLocationsInd] = yrInd - (1 if not onLeftEdge else 0)
        #     self.yrInds[angularLocationsInd] = yrInd
        #     self.ysInds[angularLocationsInd] = yrInd + (1 if not onRightEdge else 0)
        #
        #     # X terms.
        #     xrInd = yrInd + (upwardShift if not topRow else 0)
        #     self.xrInds[angularLocationsInd] = xrInd
        #
        #     # Z terms.
        #     zrInd = yrInd + (downwardShift if not onLeftEdge else 0)
        #     self.zrInds[angularLocationsInd] = zrInd

        # Initialize the neural layers.
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()
        self.activationFunction = nn.ModuleList()

    def getGivensAngles(self, layerInd):
        return torch.pi * torch.tanh(self.givensRotationParams[layerInd]) / 2  # [-pi/2, pi/2], Dim: numSignals, numParams

    @staticmethod
    def getInverseAngleParams(givensAngles):
        return torch.atanh(2 * givensAngles / torch.pi)

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
        scalingFactors = self.getJacobianScalar().flatten()  # Dim: numSignals

        return givensAngles, scalingFactors

    # ------------------------------------------------------------ #

    def getAllLinearParams(self):
        allGivensAngles, allScaleFactors = [], []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles, scalingFactors = self.getLinearParams(layerInd)
                allScaleFactors.append(scalingFactors.unsqueeze(-1).detach().cpu().numpy())
                allGivensAngles.append(givensAngles.detach().cpu().numpy())
            allGivensAngles = np.asarray(allGivensAngles)
            allScaleFactors = np.asarray(allScaleFactors)

        return allGivensAngles, allScaleFactors

    def getNumFreeParams(self):
        allNumFreeParams = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                angularMask = 0 != self.givensRotationParams[layerInd]
                numSignalParameters = angularMask.sum(dim=-1, keepdim=True)  # Dim: numSignals, 1
                allNumFreeParams.append(numSignalParameters.detach().cpu().numpy())
                # allNumFreeParams: numLayers, numSignals, numFreeParams=1

        return allNumFreeParams

    def getFeatureParams(self):
        givensAnglesFeatureNames = self.getFeatureNames()
        allGivensAnglesFeatures = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles, scalingFactors = self.getLinearParams(layerInd)  # Dim: numSignals, numParams
                scalingFactors = scalingFactors.reshape(self.numSignals, 1)  # Dim: numSignals, numParams=1
                givensAngles = givensAngles * 180 / torch.pi  # Convert to degrees

                # Calculate the mean, variance, and range of the Givens angles.
                givensAnglesRange = givensAngles.max(dim=-1).values - givensAngles.min(dim=-1).values  # Dim: numSignals
                givensAnglesVar = givensAngles.var(dim=-1).cpu().detach().numpy()  # Dim: numSignals
                givensAnglesRange = givensAnglesRange.cpu().detach().numpy()

                # Calculate the mean, variance, and range of the scaling factors.
                scalingFactorsVar = scalingFactors.var(dim=0).cpu().detach().numpy()  # Dim: numSignals

                # Combine the features. Return dimension: numFeatures, numValues
                allGivensAnglesFeatures.append([givensAnglesVar, givensAnglesRange, scalingFactorsVar])

        return givensAnglesFeatureNames, allGivensAnglesFeatures

    @staticmethod
    def getFeatureNames():
        return ["Angular variance", "Angular range", "Scalar variance"]

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
