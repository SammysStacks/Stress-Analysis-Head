from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import torch.fft
import scipy
import torch
import os

from reversibleInterface import reversibleInterface


class reversibleConvolutionLayer(reversibleInterface):

    def __init__(self, numSignals, sequenceLength, numLayers):
        super(reversibleConvolutionLayer, self).__init__()
        # General parameters.
        self.numParams = int(sequenceLength * (sequenceLength - 1) / 2)  # The number of free parameters in the model.
        self.sequenceLength = sequenceLength  # The length of the input signal.
        self.numSignals = numSignals  # The number of signals in the input data.
        self.numLayers = numLayers  # The number of layers in the reversible linear layer.
        self.optimalForwardFirst = False  # Whether to apply the forward pass first.

        # The restricted window for the neural weights.
        upperWindowMask = torch.ones(self.sequenceLength, self.sequenceLength)
        upperWindowMask = torch.triu(upperWindowMask, diagonal=1)

        # Calculate the offsets to map positions to kernel indices
        self.rowInds, self.colInds = upperWindowMask.nonzero(as_tuple=False).T

        # Initialize the neural layers.
        self.jacobianParameter = self.initializeJacobianParams(numSignals)
        self.givensRotationParams = nn.ParameterList()

        # Create the neural layers.
        for layerInd in range(self.numLayers):
            # Create the neural weights.
            parameters = nn.Parameter(torch.randn(self.numSignals, self.numParams or 1, dtype=torch.float64))
            parameters = nn.init.uniform_(parameters, a=-0.1, b=0.1)  # Dim: numSignals, numParams

            # Store the parameters.
            self.givensRotationParams.append(parameters)
            # givensRotationParams: numLayers, numSignals, numParams

    def applySingleLayer(self, inputData, layerInd):
        # Apply the weights to the input data.
        inputData = self.applyLayer(inputData, layerInd)

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
        return nn.Parameter(torch.zeros((1, numSignals, 1))) + 0.1

    def getJacobianScalar(self):
        jacobianMatrix = 1.0 + 0.1 * torch.tanh(self.jacobianParameter)
        return jacobianMatrix

    def applyManifoldScale(self, inputData):
        scalarValues = self.getJacobianScalar().expand_as(inputData)
        if not reversibleInterface.forwardDirection: return inputData * scalarValues
        else: return inputData / scalarValues

    # ------------------------------------------------------------ #

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(_p.numel() for _p in self.parameters() if _p.requires_grad) / self.numSignals
        print(f'The model has {numParams} trainable parameters.')


if __name__ == "__main__":
    # General input parameters.
    _batchSize, _numSignals, _sequenceLength = 2, 4, 128
    _numLayers = 1

    # Initialize the parameters.
    neuralLayerClass = reversibleConvolutionLayer(numSignals=_numSignals, sequenceLength=_sequenceLength, numLayers=_numLayers)
    healthProfile = torch.randn(_batchSize, _numSignals, _sequenceLength, dtype=torch.float64)

    # Normalize the health profile.
    healthProfile = healthProfile - healthProfile.min(dim=-1, keepdim=True).values
    healthProfile = healthProfile / healthProfile.max(dim=-1, keepdim=True).values
    healthProfile = healthProfile * 2 - 1

    # Perform the forward and inverse pass.
    _forwardData, _reconstructedData = neuralLayerClass.checkReconstruction(healthProfile, atol=1e-6, numLayers=1, plotResults=True)
    neuralLayerClass.printParams()

    # Calculate the Lipschitz constant.
    ratio = (_forwardData.norm(dim=-1) / healthProfile.norm(dim=-1)).view(-1).detach().numpy()
    print("Jacobian scalar:", neuralLayerClass.getJacobianScalar().mean().item())
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
    plt.show()
