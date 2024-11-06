import math

import torch
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


def getActivationMethod(activationMethod):
    if activationMethod == 'Tanhshrink':
        activationFunction = nn.Tanhshrink()
    elif activationMethod.startswith('none'):
        activationFunction = nn.Identity()
    elif activationMethod.startswith('boundedExp'):
        nonLinearityRegion = int(activationMethod.split('_')[2]) if '_' in activationMethod else 2
        topExponent = int(activationMethod.split('_')[1]) if '_' in activationMethod else 0
        activationFunction = boundedExp(decayConstant=topExponent, nonLinearityRegion=nonLinearityRegion)
    elif activationMethod.startswith('reversibleLinearSoftSign'):
        invertedActivation = activationMethod.split('_')[1] == "True"
        activationFunction = reversibleLinearSoftSign(invertedActivation=invertedActivation)
    elif activationMethod.startswith('boundedS'):
        invertedActivation = activationMethod.split('_')[1] == "True"
        activationFunction = boundedS(invertedActivation=invertedActivation)
    elif activationMethod == 'PReLU':
        activationFunction = nn.PReLU()
    elif activationMethod == 'selu':
        activationFunction = nn.SELU()
    elif activationMethod == 'gelu':
        activationFunction = nn.GELU()
    elif activationMethod == 'relu':
        activationFunction = nn.ReLU()
    else: raise ValueError("Activation type must be in ['Tanhshrink', 'none', 'boundedExp', 'reversibleLinearSoftSign', 'boundedS', 'PReLU', 'selu', 'gelu', 'relu']")

    return activationFunction


class reversibleLinearSoftSign(reversibleInterface):
    def __init__(self, invertedActivation=False, nonLinearRegion=2, infiniteBound=1):
        super(reversibleLinearSoftSign, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.nonLinearRegion = nonLinearRegion  # Corresponds to `r` in the equation
        self.infiniteBound = infiniteBound  # Corresponds to `a` in the equation
        self.tolerance = 1e-20  # Tolerance for numerical stability

        # Assert the validity of the inputs.
        assert 0 < self.infiniteBound <= 1, "The magnitude of the inf bound has a domain of (0, 1] to ensure a stable convergence."
        assert 0 < self.nonLinearRegion, "The inversion point must be positive to ensure a stable convergence."

    def forward(self, x):
        if self.forwardDirection != self.invertedActivation: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (|x| / r))
        absX = x.abs()  # Not ideal for backpropagation, but necessary for the non-linearity term

        # Calculate the non-linearity term
        if self.infiniteBound != 1: nonLinearityTerm = 1 + (self.infiniteBound / (1 - self.infiniteBound)) * (absX / self.nonLinearRegion)
        else: nonLinearityTerm = 1 + (absX / self.nonLinearRegion)

        # Calculate the output.
        y = self.infiniteBound * x + x / nonLinearityTerm

        return y

    def inversePass(self, y):
        signY = torch.nn.functional.hardtanh(y, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance

        if self.infiniteBound != 1:
            absY = y*signY
            # Calculate the non-linearity term
            squareRootTerm = ((self.infiniteBound ** 2 - 1) ** 2 * self.nonLinearRegion ** 2
                              - 2 * self.infiniteBound * self.nonLinearRegion * (self.infiniteBound - 1) ** 2 * absY
                              + self.infiniteBound ** 2 * y ** 2).sqrt()
            signDependentTerm = (squareRootTerm + (self.infiniteBound ** 2 - 1) * self.nonLinearRegion) / (2 * self.infiniteBound ** 2)
            signDependentTerm = signDependentTerm * signY

            # Combine terms, applying sign(x) for the final output
            x = signDependentTerm + y / (2 * self.infiniteBound)
        else:
            # Calculate the non-linearity term
            x = signY * ((4 * self.nonLinearRegion ** 2 + y.pow(2)).sqrt() - 2 * self.nonLinearRegion) / 2 + y / 2

        return x

class boundedS(reversibleInterface):
    def __init__(self, invertedActivation=False, inversionPoint=1):
        super(boundedS, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.inversionPoint = inversionPoint  # Corresponds to `r` in the equation
        self.tolerance = 1e-10  # Tolerance for numerical stability

        # Assert the validity of the inputs.
        assert 0 < self.inversionPoint, "The inversion point must be positive to ensure a stable convergence."

    def forward(self, x):
        if self.forwardDirection != self.invertedActivation: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (xx / rr))
        return x + x / (1 + (x / self.inversionPoint).pow(2))

    def inversePass(self, y):
        y2 = y ** 2  # Precompute y squared
        y3 = y ** 3  # Precompute y cubed
        y4 = y ** 4  # Precompute y quad

        if self.inversionPoint != 1:
            # Precompute r and r squared.
            r, r2 = self.inversionPoint, self.inversionPoint ** 2

            # Compute the cube root term
            N = 9 * r2 * y + 3 * torch.sqrt(96 * r ** 6 - 39 * r2 ** 2 * y2 + 12 * r2 * y4) + 2 * y3
            signN = torch.nn.functional.hardtanh(N, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
            cube_root_term = signN * torch.abs(N).pow(1 / 3)

            # Compute the x value
            x = (2 ** (2 / 3) * cube_root_term - 2 ** (4 / 3) * (6 * r2 - y2) / cube_root_term + 2 * y) / 6
        else:
            # Compute N
            N = 9 * y + 2 * y3 + 3 * (3*(32 - 13 * y2 + 4 * y4)).sqrt()
            signN = torch.nn.functional.hardtanh(N, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
            cube_root_term = signN * torch.abs(N).pow(1 / 3)

            # Compute x
            x = y / 3 - (2 ** (1 / 3) * (6 - y2)) / (3 * cube_root_term) + cube_root_term / (3 * (2 ** (1 / 3)))

        return x


class boundedExp(nn.Module):
    def __init__(self, decayConstant=0, nonLinearityRegion=2, infiniteBound=math.exp(-0.5)):
        super(boundedExp, self).__init__()
        # General parameters.
        self.nonLinearityRegion = nonLinearityRegion  # The non-linear region is mainly between [-nonLinearityRegion, nonLinearityRegion].
        self.infiniteBound = infiniteBound  # This controls how the activation converges at +/- infinity. The convergence is equal to inputValue*infiniteBound.
        self.decayConstant = decayConstant  # This controls the non-linearity of the data close to 0. Larger values make the activation more linear. Recommended to be 0 or 1. After 1, the activation becomes linear near 0.

        # Assert the validity of the inputs.
        assert isinstance(self.decayConstant, int), f"The decayConstant must be an integer to ensure a continuous activation, but got {type(self.decayConstant).__name__}"
        assert 0 < abs(self.infiniteBound) <= 1, "The magnitude of the inf bound has a domain of (0, 1] to ensure a stable convergence."
        assert 0 < self.nonLinearityRegion, "The non-linearity region must be positive, as negatives are redundant and 0 is linear."
        assert 0 <= self.decayConstant, "The decayConstant must be greater than 0 for the activation function to be continuous."

    def forward(self, x):
        # Calculate the exponential activation function.
        exponentialDenominator = 1 + torch.pow(x / self.nonLinearityRegion, 2 * self.decayConstant + 2)
        exponentialNumerator = torch.pow(x / self.nonLinearityRegion, 2 * self.decayConstant)
        exponentialTerm = torch.exp(exponentialNumerator / exponentialDenominator)

        # Calculate the linear term.
        linearTerm = self.infiniteBound * x

        return linearTerm * exponentialTerm


if __name__ == "__main__":
    # Test the activation functions
    data = torch.randn(2, 10, 100, dtype=torch.float64)
    data = data - data.min()
    data = data / data.max()
    data = 2 * data - 1

    # Perform the forward and inverse pass.
    activationClass = boundedS(invertedActivation=True)
    _forwardData, _reconstructedData = activationClass.checkReconstruction(data, atol=1e-6, numLayers=10)
