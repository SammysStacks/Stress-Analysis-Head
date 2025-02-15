import math
import os
import random

import numpy as np
import scipy
import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, numLayers, activationMethod):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.numParams = int(sequenceLength * (sequenceLength - 1) / 2)  # The number of free parameters in the model.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.optimalForwardFirst = False  # Whether to apply the forward pass first.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T

        # Initialize the neural layers.
        self.activationFunction = nn.ModuleList()
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(self.numSignals, self.numParams or 1, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-0.1, b=0.1)  # Dim: numSignals, numParams

            self.givensRotationParams.append(parameters)
            self.activationFunction.append(activationFunctions.getActivationMethod(activationMethod))
            # givensRotationParams: numLayers, numSignals, numParams

            # Register a gradient hook to scale the learning rate.
            parameters.register_hook(lambda grad: grad * 100)  # Double the gradient -> Doubles the effective LR

    def applySingleLayer(self, inputData, layerInd):
        # Determine the direction of the forward pass.
        performOptimalForwardFirst = self.optimalForwardFirst if layerInd % 2 == 0 else not self.optimalForwardFirst

        # Apply the weights to the input data.
        if self.activationMethod == 'none': inputData = self.applyLayer(inputData, layerInd)
        else: inputData = self.activationFunction[layerInd](inputData, lambda X: self.applyLayer(X, layerInd), forwardFirst=performOptimalForwardFirst)

        return inputData

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if self.forwardDirection: layerInd = self.numLayers - layerInd - 1
            inputData = self.applySingleLayer(inputData, layerInd)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Assert the validity of the input parameters.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert sequenceLength == self.sequenceLength, f"The sequence length is not correct: {sequenceLength}, {self.sequenceLength}"
        assert numSignals == self.numSignals, f"The number of signals is not correct: {numSignals}, {self.numSignals}"

        # Apply the neural weights to the input data.
        expA = self.getExpA(layerInd, inputData.device)  # = exp(A)
        outputData = torch.einsum('bns,nsi->bni', inputData, expA)  # Rotate: exp(A) @ f(x)
        outputData = self.applyManifoldScale(outputData)  # Scale: by jacobian
        # The inverse would be f-1(exp(-A) @ [exp(A) @ f(x)]) = X

        return outputData

    # ------------------- Rotation Methods ------------------- #

    def getExpA(self, layerInd, device):
        # Get the linear operator in the exponent.
        A = self.getA(layerInd, device)  # numSignals, sequenceLength, sequenceLength

        # Get the exponential of the linear operator.
        expA = A.matrix_exp()  # For orthogonal matrices: A.exp().inverse() = (-A).exp(); If A is Skewed Symmetric: A.exp().inverse() = A.exp().transpose()
        if self.forwardDirection: expA = expA.transpose(-2, -1)  # Take the inverse of the exponential for the forward direction.
        return expA  # exp(A)

    def getA(self, layerInd, device):
        # Gather the corresponding kernel values for each position for a skewed symmetric matrix.
        A = torch.zeros(self.numSignals, self.sequenceLength, self.sequenceLength, device=device, dtype=torch.float64)

        # Populate the Givens rotation angles.
        entriesA = self.getInfinitesimalAnglesA(layerInd)
        A[:, self.rowInds, self.colInds] = -entriesA
        A[:, self.colInds, self.rowInds] = entriesA
        # if layerInd == 1: print(entriesA[0][0].item() * 180 / 3.14159)

        return A

    def getInfinitesimalAnglesA(self, layerInd):
        return torch.pi * torch.tanh(self.givensRotationParams[layerInd]) / 2  # [-pi/2, pi/2]

    def getGivensAngles(self, layerInd):
        return self.getInfinitesimalAnglesA(layerInd)

    # ------------------- Helper Methods ------------------- #

    @staticmethod
    def initializeJacobianParams(numSignals):
        return nn.Parameter(torch.zeros((1, numSignals, 1)))

    def getJacobianScalar(self):
        jacobianMatrix = 1.0 + 0.1 * torch.tanh(self.jacobianParameter) - math.log(8*(self.numLayers + 1)) * 0.001
        return jacobianMatrix

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
        angularThresholdMin = modelConstants.userInputParams['angularThresholdMin'] * torch.pi / 180  # Convert to radians
        allNumFreeParams = []

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                angularMask = angularThresholdMin <= self.getGivensAngles(layerInd).abs()
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
                givensAnglesMedian = torch.median(givensAngles, dim=-1).values.cpu().detach().numpy()  # Dim: numSignals

                # Calculate the mean, variance, and range of the positive Givens angles.
                givensAnglesMeanABS = givensAnglesABS.mean(dim=-1).cpu().detach().numpy()  # Dim: numSignals
                givensAnglesVarABS = givensAnglesABS.var(dim=-1).cpu().detach().numpy()  # Dim: numSignals

                # Calculate the mean, variance, and range of the scaling factors.
                scalingFactorsMean = scalingFactors.mean(dim=0).cpu().detach().numpy()  # Dim: numSignals=1
                scalingFactorsVar = scalingFactors.var(dim=0).cpu().detach().numpy()  # Dim: numSignals

                # Combine the features. Return dimension: numFeatures, numValues
                allGivensAnglesFeatures.append([givensAnglesMean, givensAnglesVar, givensAnglesRange, givensAnglesMedian, givensAnglesMeanABS, givensAnglesVarABS, scalingFactorsMean, scalingFactorsVar])

        return givensAnglesFeatureNames, allGivensAnglesFeatures

    @staticmethod
    def getFeatureNames():
        return ["Angular mean", "Angular variance", "Angular range", "Angular median", "Angular abs(mean)", "Angular abs(variance)", "Scalar mean", "Scalar variance"]

    def angularThresholding(self, applyMinThresholding, applyMaxThresholding):
        # Get the angular thresholds.
        angularThresholdMin = (4 if applyMaxThresholding else modelConstants.userInputParams['angularThresholdMin']) * torch.pi / 180  # Convert to radians
        angularThresholdMax = modelConstants.userInputParams['angularThresholdMax'] * torch.pi / 180  # Convert to radians

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles = self.getGivensAngles(layerInd)

                # Apply the thresholding.
                if applyMinThresholding and 64 < self.sequenceLength: self.minThresholding(layerInd, applyMaxThresholding)
                if applyMinThresholding: self.givensRotationParams[layerInd][givensAngles.abs() < angularThresholdMin] = 0
                self.givensRotationParams[layerInd][givensAngles <= -angularThresholdMax] = -angularThresholdMax
                self.givensRotationParams[layerInd][angularThresholdMax <= givensAngles] = angularThresholdMax

    def minThresholding(self, layerInd, applyMaxThresholding):
        with torch.no_grad():
            # Sort each row by absolute value
            givensAngles = self.getGivensAngles(layerInd).clone().abs()  # Dim: numSignals, numParams
            sorted_values, sorted_indices = torch.sort(givensAngles, dim=-1)
            # sorted_values -> [0, 1, 2, 3, ...]

            # Find the threshold value per row
            percentParamsKeeping = float(200/self.numParams if applyMaxThresholding else modelConstants.userInputParams['percentParamsKeeping'])
            numAnglesThrowingAway = int((100 - percentParamsKeeping) * self.numParams / 100) - 1
            minAngleValues = sorted_values[:, numAnglesThrowingAway].unsqueeze(-1)  # Shape (numSignals, 1)

            # Zero out the values below the threshold
            self.givensRotationParams[layerInd][givensAngles < minAngleValues] = 0

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

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(_p.numel() for _p in self.parameters() if _p.requires_grad) / self.numSignals
        print(f'The model has {numParams} trainable parameters.')


if __name__ == "__main__":
    # for i in [2, 4, 8, 16, 32, 64, 128, 256]:
    # for i in [16, 32, 64, 128, 256]:
    reconstructionFlag = True

    try:
        # for layers, sequenceLength2 in [(2, 256), (2, 128), (2, 64), (2, 32), (2, 16), (2, 8), (2, 4), (2, 2)]:
        # for _layerInd, sequenceLength2 in [(1, 32), (2, 32), (3, 32), (5, 32), (5, 32), (10, 32)]:
        # for _layerInd, sequenceLength2 in [(1, 64), (2, 64), (3, 64), (5, 64), (5, 64), (10, 64)]:
        # for _layerInd, sequenceLength2 in [(1, 128), (2, 128), (3, 128), (5, 128), (5, 128), (10, 128)]:
        for _layerInd, sequenceLength2 in [(1, 256)]:
            # General parameters.
            _batchSize, _numSignals, _sequenceLength = 128, 128, sequenceLength2
            _activationMethod = 'reversibleLinearSoftSign'  # reversibleLinearSoftSign
            _numLayers = _layerInd

            # Set up the parameters.
            neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, numLayers=_numLayers, activationMethod=_activationMethod)
            healthProfile = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
            healthProfile = healthProfile - healthProfile.min(dim=-1, keepdim=True).values
            healthProfile = healthProfile / healthProfile.max(dim=-1, keepdim=True).values
            healthProfile = healthProfile * 2 - 1

            # Perform the convolution in the fourier and spatial domains.
            if reconstructionFlag: _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(healthProfile, atol=1e-6, numLayers=1, plotResults=True)
            else: _forwardData = neuralLayerClass.forward(healthProfile)
            neuralLayerClass.printParams()

            ratio = (_forwardData.norm(dim=-1) / healthProfile.norm(dim=-1)).view(-1).detach().numpy()
            if abs(ratio.mean() - 1) < 0.1: plt.hist(ratio, bins=150, alpha=0.2, label=f'len{_sequenceLength}_layers={_layerInd}', density=True)
            print("Lipschitz constant:", ratio.mean())

            # Plot the Gaussian fit
            xmin, xmax = plt.xlim()
            x_ = np.linspace(xmin, xmax, num=1000)
            mu, std = scipy.stats.norm.fit(ratio)
            p = scipy.stats.norm.pdf(x_, mu, std)
            plt.plot(x_, p, 'k', linewidth=2, label=f'Gaussian fit: mu={mu:.8f}$, sigma={std:.8f}$')

            # Figure settings.
            plt.title(f'Fin', fontsize=14)  # Increase title font size for readability
            # plt.xlim(0.98, 1.02)
            plt.legend()

            # Save the plot.
            os.makedirs('_lipshitz/', exist_ok=True)
            plt.savefig(f'_lipshitz/len{sequenceLength2}_layers={_numLayers}')
            plt.show()
    except Exception as e: print(f"Error: {e}")
