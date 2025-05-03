import os

import numpy as np
import scipy
import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayerInterface import reversibleLieLayerInterface


class reversibleLieLayer(reversibleLieLayerInterface):

    def __init__(self, numSignals, sequenceLength, numLayers, activationMethod):
        super(reversibleLieLayer, self).__init__(numSignals, sequenceLength, numLayers, activationMethod)
        self.initialMaxGivensAngle = self.getInverseAngleParams(torch.as_tensor(3 / sequenceLength).sqrt() * torch.pi / 180)
        self.identityMatrix = torch.eye(self.sequenceLength, dtype=torch.float64)

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = torch.randn(self.numSignals, self.numParams or 1, dtype=torch.float64)
            parameters = nn.init.uniform_(parameters, a=-self.initialMaxGivensAngle, b=self.initialMaxGivensAngle)
            # parameters: numSignals, numParams

            # Store the parameters.
            self.activationFunction.append(activationFunctions.getActivationMethod(activationMethod))
            self.givensRotationParams.append(nn.Parameter(parameters))  # givensRotationParams: numLayers, numSignals, numParams

    # ------------------- Main Sections ------------------- #

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if self.forwardDirection: layerInd = self.numLayers - layerInd - 1
            inputData = self.applySingleLayer(inputData, layerInd)

        return inputData

    def getExpS(self, layerInd):
        # Get the exponential of the linear operator.
        expS = self.matrixExp_skewSymmetric(self.getS(layerInd))  # For orthogonal matrices: A.exp().inverse() = (-A).exp(); If A is Skewed Symmetric: A.exp().inverse() = A.exp().transpose()
        if self.forwardDirection: expS = expS.transpose(-2, -1)  # Take the inverse of the exponential for the forward direction.

        return expS

    def getS(self, layerInd):
        # Get the relevant model parameters.
        entriesS = self.getGivensAngles(layerInd)

        # Gather the corresponding kernel values for each position for a skewed symmetric matrix.
        S = torch.zeros(self.numSignals, self.sequenceLength, self.sequenceLength, dtype=entriesS.dtype, device=entriesS.device)

        # Populate the Givens rotation angles.
        S[:, self.rowInds, self.colInds] = -entriesS
        S[:, self.colInds, self.rowInds] = entriesS

        return S

    def matrixExp_skewSymmetric(self, S):
        if S.size(-1) <= 512: return S.matrix_exp()
        else: return self.matrix_exp_approx(S, terms=8)

    def matrix_exp_approx(self, S, terms):
        """ Approximates the matrix exponential using a higher-order Taylor series expansion. """
        identityMatrix = self.identityMatrix.to(S.device).expand_as(S)
        result = identityMatrix + S
        term = S
        for i in range(2, terms + 1):
            term = term @ S / i
            result = result + term
        return result

    # ------------------- General Sections ------------------- #

    def applyLayer(self, inputData, layerInd):
        # Assert the validity of the input parameters.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert sequenceLength == self.sequenceLength, f"The sequence length is not correct: {sequenceLength}, {self.sequenceLength}"
        assert numSignals == self.numSignals, f"The number of signals is not correct: {numSignals}, {self.numSignals}"

        # Apply the neural weights to the input data.
        expS = self.getExpS(layerInd)  # = exp(S)
        outputData = torch.einsum('...ns,nsi->...ni', inputData, expS)  # Rotate: exp(S) @ f(x)
        outputData = self.applyManifoldScale(outputData)  # Scale: by jacobian
        # The inverse would be f-1(exp(-A) @ [exp(S) @ f(x)]) = X

        return outputData

    def applySingleLayer(self, inputData, layerInd):
        # Determine the direction of the forward pass.
        performOptimalForwardFirst = self.optimalForwardFirst if layerInd % 2 == 0 else not self.optimalForwardFirst

        # Apply the layer.
        inputData = self.applyLayer(inputData, layerInd) if self.activationMethod == 'none' \
            else self.activationFunction[layerInd](inputData, lambda X: self.applyLayer(X, layerInd), forwardFirst=performOptimalForwardFirst)

        return inputData

    # ------------------- Observational Learning ------------------- #

    @staticmethod
    def getWarmupThreshold():
        return 0.001 * torch.pi / 180

    @staticmethod
    def getMinAngularThreshold(epoch, sharedLayer=False):
        if epoch <= modelConstants.numWarmupEpochs: return reversibleLieLayer.getWarmupThreshold()
        relativeEpoch = epoch - modelConstants.numWarmupEpochs

        # Get the minimum angular threshold.
        minThresholdStep = modelConstants.userInputParams['minThresholdStep']
        minAngularThreshold = modelConstants.userInputParams['minAngularThreshold'] if not sharedLayer else 0.01
        minAngularThreshold = min(minAngularThreshold, (relativeEpoch**1.5) * minThresholdStep) * torch.pi/180

        return minAngularThreshold

    def angularThresholding(self, epoch, sharedLayer):
        with torch.no_grad():
            # Get the angular maximum thresholds.
            maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold'] * torch.pi/180  # Convert to radians
            maxAngularParam = self.getInverseAngleParams(maxAngularThreshold)

            # Get the angular minimum thresholds.
            if epoch <= modelConstants.numWarmupEpochs: minAngularThreshold = self.getInverseAngleParams(self.getWarmupThreshold())
            else: minAngularThreshold = self.getInverseAngleParams(self.getMinAngularThreshold(epoch, sharedLayer))

            for layerInd in range(self.numLayers):
                givensAngles = self.getGivensAngles(layerInd)  # Dim: numSignals, numParams
                self.givensRotationParams[layerInd][givensAngles <= -maxAngularThreshold] = -maxAngularParam
                self.givensRotationParams[layerInd][maxAngularThreshold <= givensAngles] = maxAngularParam
                self.givensRotationParams[layerInd][givensAngles.abs() < minAngularThreshold] = 0

    # ------------------------------------------------------------ #

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(_p.numel() for _p in self.parameters() if _p.requires_grad) / self.numSignals
        print(f'The model has {numParams} trainable parameters.')


if __name__ == "__main__":
    # for i in [2, 4, 8, 16, 32, 64, 128, 256]:
    # for i in [16, 32, 64, 128, 256]:
    modelConstants.userInputParams['submodel'] = modelConstants.signalEncoderModel
    modelConstants.userInputParams['minAngularThreshold'] = 0.1
    modelConstants.userInputParams['maxAngularThreshold'] = 45

    # for layers, sequenceLength2 in [(2, 256), (2, 128), (2, 64), (2, 32), (2, 16), (2, 8), (2, 4), (2, 2)]:
    # for _layerInd, sequenceLength2 in [(1, 32), (2, 32), (3, 32), (5, 32), (5, 32), (10, 32)]:
    # for _layerInd, sequenceLength2 in [(1, 64), (2, 64), (3, 64), (5, 64), (5, 64), (10, 64)]:
    # for _layerInd, sequenceLength2 in [(1, 128), (2, 128), (3, 128), (5, 128), (5, 128), (10, 128)]:
    # for _layerInd, sequenceLength2 in [(1, 8), (1, 16), (1, 32), (1, 64), (1, 128), (1, 256), (1, 512)]:
    for _layerInd, sequenceLength2 in [(1, 256)]:
        # General parameters.
        _batchSize, _numSignals, _sequenceLength = 256, 128, sequenceLength2
        _activationMethod = 'reversibleLinearSoftSign'  # reversibleLinearSoftSign
        _numLayers = _layerInd

        # Set up the parameters.
        neuralLayerClass = reversibleLieLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, numLayers=_numLayers, activationMethod=_activationMethod)
        neuralLayerClass = neuralLayerClass.double()

        # Generate the health profile.
        sequence = torch.arange(_sequenceLength, dtype=torch.float64)
        healthProfile = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
        # healthProfile = healthProfile/4 + (2*sequence).sin() + sequence.cos()
        healthProfile = healthProfile - healthProfile.min(dim=-1, keepdim=True).values
        healthProfile = healthProfile / healthProfile.max(dim=-1, keepdim=True).values
        healthProfile = healthProfile * 2 - 1

        # Check the reconstruction forwards and backwards.
        _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(healthProfile, atol=1e-6, numLayers=1, plotResults=False, title=f'len{_sequenceLength}_layers={_numLayers}')
        neuralLayerClass.printParams()

        ratio = (_forwardData.norm(dim=-1) / healthProfile.norm(dim=-1)).view(-1).detach().numpy().astype(np.float32)
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
        plt.legend()

        # Save the plot.
        os.makedirs('_lipshitz/', exist_ok=True)
        plt.savefig(f'_lipshitz/len{sequenceLength2}_layers={_numLayers}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
