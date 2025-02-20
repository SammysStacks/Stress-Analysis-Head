import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleLieLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, numLayers, activationMethod):
        super(reversibleLieLayer, self).__init__()
        # General parameters.
        self.numParams = int(sequenceLength * (sequenceLength - 1) / 2)  # The number of free parameters in the model.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.optimalForwardFirst = False  # Whether to apply the forward pass first.

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
        coupling = 1/32; self.alpha = 1/4 - coupling
        self.numDegreesShifting = 1
        self.beta = 1/2 + 2*coupling

        self.xwInds, self.zwInds, self.yzInds, self.xyInds = [], [], [], []
        for angularLocationsInd in self.angularLocationsInds:
            i, j = self.rowInds[angularLocationsInd], self.colInds[angularLocationsInd]
            if j - i == 1: continue  # Skip the first upper diagonal elements

            nextRowLength = self.sequenceLength - i - 2
            # Get the four sub-rotation indices: [X, Y, Z, W]
            self.zwInds.append(angularLocationsInd + 2 * nextRowLength - 1)
            self.yzInds.append(angularLocationsInd + nextRowLength - 1)
            self.xyInds.append(angularLocationsInd - 2)
            self.xwInds.append(angularLocationsInd)

        # Initialize the neural layers.
        self.activationFunction = nn.ModuleList()
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(self.numSignals, self.numParams or 1, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-0.1, b=0.1)  # Dim: numSignals, numParams
            parameters[:, self.angularMaskInds].fill_(0)  # Apply checkerboard thresholding.

            # Store the parameters.
            self.activationFunction.append(activationFunctions.getActivationMethod(activationMethod))
            self.givensRotationParams.append(parameters)  # givensRotationParams: numLayers, numSignals, numParams
            self.applyAngularBias(layerInd)  # Inject bias towards banded structure.

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
        expS = self.getExpS(layerInd, inputData.device)  # = exp(S)
        outputData = torch.einsum('bns,nsi->bni', inputData, expS)  # Rotate: exp(S) @ f(x)
        outputData = self.applyManifoldScale(outputData)  # Scale: by jacobian
        # The inverse would be f-1(exp(-A) @ [exp(S) @ f(x)]) = X

        return outputData

    # ------------------- Rotation Methods ------------------- #

    def getExpS(self, layerInd, device):
        # Get the linear operator in the exponent.
        S = self.getS(layerInd, device)  # numSignals, sequenceLength, sequenceLength

        # Get the exponential of the linear operator.
        expS = S.matrix_exp()  # For orthogonal matrices: A.exp().inverse() = (-A).exp(); If A is Skewed Symmetric: A.exp().inverse() = A.exp().transpose()
        if self.forwardDirection: expS = expS.transpose(-2, -1)  # Take the inverse of the exponential for the forward direction.
        return expS  # exp(S)

    def getS(self, layerInd, device):
        # Gather the corresponding kernel values for each position for a skewed symmetric matrix.
        S = torch.zeros(self.numSignals, self.sequenceLength, self.sequenceLength, device=device, dtype=torch.float64)

        # Populate the Givens rotation angles.
        entriesS = self.getInfinitesimalAnglesA(layerInd)
        S[:, self.rowInds, self.colInds] = -entriesS
        S[:, self.colInds, self.rowInds] = entriesS

        return S

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

    def applyAngularBias(self, layerInd):
        with (torch.no_grad()):
            # Create update matrix.
            angularUpdateMatrix = torch.zeros_like(self.givensRotationParams[layerInd])
            countMatrix = torch.zeros_like(self.givensRotationParams[layerInd])
            # angularUpdateMatrix: numSignals, numParams

            # Update the four angles in the 4D sub-rotation matrix: [X, Y, Z, W]
            angularUpdateValue = self.givensRotationParams[layerInd][:, self.xwInds].sign() * self.numDegreesShifting * torch.pi / 180  # Dim: numSignals, numParams
            angularUpdateMatrix[:, self.xwInds] -= angularUpdateValue  # XW
            angularUpdateMatrix[:, self.xyInds] += angularUpdateValue*self.alpha  # XY
            angularUpdateMatrix[:, self.yzInds] -= angularUpdateValue*self.beta  # YZ
            angularUpdateMatrix[:, self.zwInds] += angularUpdateValue*self.alpha  # ZW

            # Average the update matrix.
            countMatrix[:, self.xwInds] += 1
            countMatrix[:, self.xyInds] += 1
            countMatrix[:, self.yzInds] += 1
            countMatrix[:, self.zwInds] += 1
            countMatrix[countMatrix == 0] = 1

            # Apply a gradient mask.
            angularUpdateMatrix[:, self.angularLocationsInds] *= (self.colInds[self.angularLocationsInds] - self.rowInds[self.angularLocationsInds]
                                                                  ).abs().to(angularUpdateMatrix.device) / self.sequenceLength

            # import matplotlib.pyplot as plt
            # S = torch.zeros((self.numSignals, self.sequenceLength, self.sequenceLength), dtype=torch.float64)
            # S[:, self.rowInds, self.colInds] = angularUpdateMatrix
            # S[:, self.colInds, self.rowInds] = -angularUpdateMatrix
            # plt.imshow(S[0].cpu().detach().numpy(), cmap='plasma'); plt.colorbar(); plt.show()
            # print(S[0].cpu().detach().numpy()[0])

            # Apply the update.
            self.givensRotationParams[layerInd].add_(angularUpdateMatrix / countMatrix)

    def angularThresholding(self, applyMaxThresholding):
        # Get the angular thresholds.
        minAngularThreshold = modelConstants.userInputParams['finalMinAngularThreshold' if applyMaxThresholding else 'minAngularThreshold'] * torch.pi / 180  # Convert to radians
        maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold'] * torch.pi / 180  # Convert to radians

        with torch.no_grad():
            for layerInd in range(self.numLayers):
                givensAngles = self.getGivensAngles(layerInd)

                # Apply the maximum thresholding.
                self.givensRotationParams[layerInd][givensAngles <= -maxAngularThreshold].fill_(-maxAngularThreshold)
                self.givensRotationParams[layerInd][maxAngularThreshold <= givensAngles].fill_(maxAngularThreshold)

                # Apply the minimum thresholding.
                self.givensRotationParams[layerInd][givensAngles.abs() < minAngularThreshold].fill_(0)

                # Removing (approximate) geometric symmetries.
                self.givensRotationParams[layerInd][:, self.angularMaskInds].fill_(0)
                if 64 < self.sequenceLength: self.percentParamThresholding(layerInd)
                self.applyAngularBias(layerInd)  # Inject bias towards banded structure.

    def percentParamThresholding(self, layerInd):
        with torch.no_grad():
            # Sort each row by absolute value
            givensAngles = self.getGivensAngles(layerInd).clone().abs()  # Dim: numSignals, numParams
            sorted_values, sorted_indices = torch.sort(givensAngles, dim=-1)
            # sorted_values -> [0, 1, 2, 3, ...]

            # Get the thresholding information.
            percentParamsKeeping = float(modelConstants.userInputParams['percentParamsKeeping'])

            # Find the threshold angles.
            numAnglesThrowingAway = int((100 - percentParamsKeeping) * self.numParams / 100) - 1
            minAngleValues = sorted_values[:, numAnglesThrowingAway].unsqueeze(-1)  # Shape (numSignals, 1)

            # Zero out the values below the threshold
            self.givensRotationParams[layerInd][givensAngles < minAngleValues].fill_(0)

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

    # for layers, sequenceLength2 in [(2, 256), (2, 128), (2, 64), (2, 32), (2, 16), (2, 8), (2, 4), (2, 2)]:
    # for _layerInd, sequenceLength2 in [(1, 32), (2, 32), (3, 32), (5, 32), (5, 32), (10, 32)]:
    # for _layerInd, sequenceLength2 in [(1, 64), (2, 64), (3, 64), (5, 64), (5, 64), (10, 64)]:
    # for _layerInd, sequenceLength2 in [(1, 128), (2, 128), (3, 128), (5, 128), (5, 128), (10, 128)]:
    for _layerInd, sequenceLength2 in [(4, 256)]:
        # General parameters.
        _batchSize, _numSignals, _sequenceLength = 128, 64, sequenceLength2
        _activationMethod = 'reversibleLinearSoftSign'  # reversibleLinearSoftSign
        _numLayers = _layerInd

        # Set up the parameters.
        neuralLayerClass = reversibleLieLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, numLayers=_numLayers, activationMethod=_activationMethod)
        healthProfile = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
        healthProfile = healthProfile - healthProfile.min(dim=-1, keepdim=True).values
        healthProfile = healthProfile / healthProfile.max(dim=-1, keepdim=True).values
        healthProfile = healthProfile * 2 - 1

        # Perform the convolution in the fourier and spatial domains.
        _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(healthProfile, atol=1e-6, numLayers=1, plotResults=True)
        neuralLayerClass.printParams()

        # ratio = (_forwardData.norm(dim=-1) / healthProfile.norm(dim=-1)).view(-1).detach().numpy()
        # if abs(ratio.mean() - 1) < 0.1: plt.hist(ratio, bins=150, alpha=0.2, label=f'len{_sequenceLength}_layers={_layerInd}', density=True)
        # print("Lipschitz constant:", ratio.mean())
        #
        # # Plot the Gaussian fit
        # xmin, xmax = plt.xlim()
        # x_ = np.linspace(xmin, xmax, num=1000)
        # mu, std = scipy.stats.norm.fit(ratio)
        # p = scipy.stats.norm.pdf(x_, mu, std)
        # plt.plot(x_, p, 'k', linewidth=2, label=f'Gaussian fit: mu={mu:.8f}$, sigma={std:.8f}$')
        #
        # # Figure settings.
        # plt.title(f'Fin', fontsize=14)  # Increase title font size for readability
        # # plt.xlim(0.98, 1.02)
        # plt.legend()
        #
        # # Save the plot.
        # os.makedirs('_lipshitz/', exist_ok=True)
        # plt.savefig(f'_lipshitz/len{sequenceLength2}_layers={_numLayers}')
        # plt.show()
