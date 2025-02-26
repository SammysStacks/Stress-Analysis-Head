import torch.nn as nn
import numpy as np
import torch.fft
import torch

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

        # Create a mask for the angles.
        angleMask = torch.ones(self.sequenceLength, self.sequenceLength, dtype=torch.int32)
        angleMask[::2, ::2] = -1; angleMask[1::2, 1::2] = -1
        angleMask = torch.triu(angleMask, diagonal=1)
        angleMask = angleMask[angleMask != 0].flatten()
        angleMask[angleMask == 1] = 0

        # Calculate the offsets to map positions to kernel indices
        self.angularLocationsInds = (angleMask == 0).nonzero(as_tuple=False).T[0]
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T
        self.angularMaskInds = angleMask.nonzero(as_tuple=False).T[0]

        # Define angular update parameters.
        self.angularShiftingPercent = modelConstants.userInputParams['angularShiftingPercent']
        self.decayFactorCheckerboard, self.decayFactorThreshold = 0, 1/8
        self.alpha, self.beta, self.gamma = 1, 1, 1

        # Get the four sub-rotation indices: [X, Y, Z, W]
        self.xwInds, self.xzInds, self.yWInds = [], [], []
        self.zwInds, self.yzInds, self.xyInds = [], [], []
        for angularLocationsInd in self.angularLocationsInds:
            i, j = self.rowInds[angularLocationsInd], self.colInds[angularLocationsInd]
            if j - i == 1: continue  # Skip the first upper diagonal elements
            nextRowLength = self.sequenceLength - i - 2

            # Static terms.
            self.yzInds.append(angularLocationsInd + nextRowLength - 1)
            self.xwInds.append(angularLocationsInd)

            # Alpha terms.
            self.zwInds.append(angularLocationsInd + 2 * nextRowLength - 1)
            self.xyInds.append(angularLocationsInd - 2)

            # Coupling terms.
            self.xzInds.append(angularLocationsInd - 1)
            self.yWInds.append(angularLocationsInd + nextRowLength)

        # Initialize the neural layers.
        self.activationFunction = nn.ModuleList()
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()
        self.numShiftedRotations = []

    def getGivensAngles(self, layerInd):
        return torch.pi * torch.tanh(self.givensRotationParams[layerInd]) / 2  # [-pi/2, pi/2]

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

    def getNumFreeParams(self, applyMaxThresholding):
        minAngularThreshold = modelConstants.userInputParams['finalMinAngularThreshold' if applyMaxThresholding else 'minAngularThreshold'] * torch.pi / 180  # Convert to radians
        allNumFreeParams = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                angularMask = minAngularThreshold <= self.getGivensAngles(layerInd).abs()
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
                givensAnglesABS = givensAngles.abs()

                # Calculate the mean, variance, and range of the Givens angles.
                givensAnglesRange = givensAngles.max(dim=-1).values - givensAngles.min(dim=-1).values  # Dim: numSignals
                givensAnglesMean = givensAngles.mean(dim=-1).cpu().detach().numpy()  # Dim: numSignals
                givensAnglesVar = givensAngles.var(dim=-1).cpu().detach().numpy()  # Dim: numSignals
                givensAnglesRange = givensAnglesRange.cpu().detach().numpy()

                # Calculate the mean, variance, and range of the positive Givens angles.
                givensAnglesMeanABS = givensAnglesABS.mean(dim=-1).cpu().detach().numpy()  # Dim: numSignals
                givensAnglesVarABS = givensAnglesABS.var(dim=-1).cpu().detach().numpy()  # Dim: numSignals

                # Calculate the mean, variance, and range of the scaling factors.
                scalingFactorsVar = scalingFactors.var(dim=0).cpu().detach().numpy()  # Dim: numSignals

                # Combine the features. Return dimension: numFeatures, numValues
                allGivensAnglesFeatures.append([givensAnglesMean, givensAnglesVar, givensAnglesRange, givensAnglesMeanABS, givensAnglesVarABS, scalingFactorsVar])

        return givensAnglesFeatureNames, allGivensAnglesFeatures

    @staticmethod
    def getFeatureNames():
        return ["Angular mean", "Angular variance", "Angular range", "Angular abs(mean)", "Angular abs(variance)", "Scalar variance"]

    def getAllActivationParams(self):
        allActivationParams = []
        with torch.no_grad():
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
