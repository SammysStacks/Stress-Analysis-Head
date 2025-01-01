import os

import numpy as np
import torch
import torch.fft
import torch.nn as nn
from matplotlib import pyplot as plt
import scipy

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, kernelSize, numLayers, activationMethod):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.activationMethod = activationMethod  # The activation method to use.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.kernelSize = kernelSize  # The restricted window for the neural weights.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength, dtype=torch.float64)
        if self.sequenceLength != self.kernelSize: upperWindowMask = torch.tril(upperWindowMask, diagonal=self.kernelSize//2)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T
        self.kernelInds = self.rowInds - self.colInds + self.kernelSize // 2  # Adjust for the kernel center

        # Assert the validity of the input parameters.
        assert 1 <= self.kernelSize // 2 <= sequenceLength - 1, f"The kernel size must be less than the sequence length: {self.kernelSize}, {self.sequenceLength}"
        assert self.kernelInds.max() == self.kernelSize // 2 - 1, f"The kernel indices are not valid: {self.kernelInds.max()}"
        assert self.kernelInds.min() == 0, f"The kernel indices are not valid: {self.kernelInds.min()}"

        # Initialize the neural layers.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod)
        self.linearOperators = nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(numSignals, self.kernelSize//2 or 1, dtype=torch.float64))
            parameters = nn.init.kaiming_uniform_(parameters)
            self.linearOperators.append(parameters)

    def forward(self, inputData):
        for layerInd in range(self.numLayers):
            if not self.forwardDirection: layerInd = self.numLayers - layerInd - 1

            # Apply the weights to the input data.
            if self.activationMethod == 'none': inputData = self.applyLayer(inputData, layerInd)
            else: inputData = self.activationFunction(inputData, lambda X: self.applyLayer(X, layerInd), forwardFirst=False)

        return inputData

    def applyLayer(self, inputData, layerInd):
        # Assert the validity of the input parameters.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert sequenceLength == self.sequenceLength, f"The sequence length is not correct: {sequenceLength}, {self.sequenceLength}"
        assert numSignals == self.numSignals, f"The number of signals is not correct: {numSignals}, {self.numSignals}"

        # Apply the neural weights to the input data.
        neuralWeights = self.getTransformationMatrix(layerInd, inputData.device)  # = exp(A)
        outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)  # -> exp(A) @ f(x)
        # THe inverse would be f-1[exp(A) @ f(x)]

        return outputData

    def getTransformationMatrix(self, layerInd, device):
        # Unpack the dimensions.
        neuralWeights = torch.zeros(self.numSignals, self.sequenceLength, self.sequenceLength, dtype=torch.float64, device=device)
        # neuralWeight: numSignals, sequenceLength, sequenceLength

        # Gather the corresponding kernel values for each position for a skewed symmetric matrix.
        neuralWeights[:, self.rowInds, self.colInds] = -self.linearOperators[layerInd][:, self.kernelInds]
        neuralWeights[:, self.colInds, self.rowInds] = self.linearOperators[layerInd][:, self.kernelInds]
        # neuralWeight: numSignals, sequenceLength, sequenceLength

        # Create an orthogonal matrix.
        neuralWeights = neuralWeights.matrix_exp()
        if self.forwardDirection: neuralWeights = neuralWeights.transpose(-2, -1)  # Ensure the neural weights are symmetric.
        # For orthogonal matrices: A.exp().inverse() = A.exp().transpose() = (-A).exp()

        return neuralWeights
    
    def getAllEigenvalues(self, device):
        allEigenvalues = np.zeros(shape=(self.numLayers, self.numSignals, self.sequenceLength), dtype=np.complex128)
        for layerInd in range(self.numLayers): allEigenvalues[layerInd] = self.getEigenvalues(layerInd, device)
        return allEigenvalues

    def getEigenvalues(self, layerInd, device):
        neuralWeights = self.getTransformationMatrix(layerInd, device).detach()
        return torch.linalg.eigvals(neuralWeights).detach().cpu().numpy()

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(_p.numel() for _p in self.parameters() if _p.requires_grad) / self.numSignals
        print(f'The model has {numParams} trainable parameters.')


if __name__ == "__main__":
    # for i in [2, 4, 8, 16, 32, 64, 128, 256]:
    # for i in [16, 32, 64, 128, 256]:
    reconstructionFlag = False

    try:
        # for layers, sequenceLength2 in [(2, 256), (2, 128), (2, 64), (2, 32), (2, 16), (2, 8), (2, 4), (2, 2)]:
        for _layerInd, sequenceLength2 in [(1, 128)]:
            # General parameters.
            _batchSize, _numSignals, _sequenceLength = 256, 256, sequenceLength2
            _kernelSize = 2*_sequenceLength - 1
            _activationMethod = 'reversibleLinearSoftSign'  # reversibleLinearSoftSign
            _numLayers = _layerInd

            # Set up the parameters.
            neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, kernelSize=_kernelSize, numLayers=_numLayers, activationMethod=_activationMethod)
            healthProfile = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)
            healthProfile = healthProfile - healthProfile.mean(dim=-1, keepdim=True)
            healthProfile = healthProfile / healthProfile.std(dim=-1, keepdim=True)
            healthProfile = healthProfile / 3

            # Perform the convolution in the fourier and spatial domains.
            if reconstructionFlag: _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(healthProfile, atol=1e-6, numLayers=1, plotResults=False)
            else: _forwardData = neuralLayerClass.forward(healthProfile)
            neuralLayerClass.printParams()

            ratio = (_forwardData.norm(dim=-1) / healthProfile.norm(dim=-1)).view(-1).detach().numpy()
            if abs(ratio.mean() - 1) < 0.1: plt.hist(ratio, bins=150, alpha=0.2, label=f'len{_sequenceLength}_layers={_layerInd}', density=True)
            print(ratio.mean())

            # Plot the Gaussian fit
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, num=1000)
            mu, std = scipy.stats.norm.fit(ratio)
            p = scipy.stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian fit: mu={mu:.8f}$, sigma={std:.8f}$')

            # Figure settings.
            plt.title(f'Fin', fontsize=14)  # Increase title font size for readability
            plt.xlim(0.98, 1.02)
            plt.legend()

            # Save the plot.
            os.makedirs('_lipshitz/', exist_ok=True)
            plt.savefig(f'_lipshitz/len{sequenceLength2}_layers={_numLayers}')
            plt.show()
    except Exception as e: print(f"Error: {e}")
