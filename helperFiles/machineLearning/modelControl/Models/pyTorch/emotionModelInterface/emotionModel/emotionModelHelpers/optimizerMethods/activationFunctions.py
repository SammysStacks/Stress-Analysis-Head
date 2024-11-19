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
        activationFunction = reversibleLinearSoftSign()
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
    def __init__(self, inversionPoint=2):
        super(reversibleLinearSoftSign, self).__init__()
        self.inversionPoint = inversionPoint  # The point at which the activation inverts. Higher values increase the non-linearity and decrease the final magnitude.
        self.tolerance = 1e-20  # Tolerance for numerical stability

        # If the infiniteBound term is not provided, use the r that makes y = x = 1.
        self.linearity = 2 / (1 + self.inversionPoint)  # Corresponds to `r` in the equation
        self.infiniteBound = 1 - 1/((1 + self.inversionPoint)*self.linearity)  # This controls how the activation converges at +/- infinity; Ex: 0.5, 13/21, 33/49

        # Assert the validity of the inputs.
        assert 1 <= self.inversionPoint, "The inversion point must be greater than 1 to ensure a stable convergence."
        assert self.infiniteBound == 0.5, "The infinite bound term must be 0.5 to ensure a stable convergence!!"
        # Notes: The linearity term must be 1 if the inversion point is 1 to ensure a stable convergence.
        # Notes: The inversion point must be greater than 1 to ensure a stable convergence.

    def forward(self, x, linearModel):
        x = self.forwardPass(x)  # Increase the signal.
        x = linearModel(x)  # Learn in the scaled domain.
        x = self.inversePass(x)  # Decrease the signal.

        return x

    def forwardPass(self, x):
        return self.infiniteBound*x + x / (1 + x.abs()) / self.linearity  # f(x) = x + x / (1 + |x|) / r

    def inversePass(self, y):
        # Prepare the terms for the inverse pass.
        signY = torch.nn.functional.hardtanh(y, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
        r, a = self.linearity, self.infiniteBound  # The linearity and infinite bound terms

        sqrtTerm = ((r*a)**2 + 2*a*r*(1 + signY*y*r) + (r*y - signY).pow(2)) / (r*a)**2
        x = signY*(sqrtTerm.sqrt() - 1)/2 - signY / (2*a*r) + y / (2*a)

        return x


class boundedS(reversibleInterface):
    def __init__(self, invertedActivation=False, linearity=1):
        super(boundedS, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.linearity = linearity  # Corresponds to `r` in the equation
        self.tolerance = 1e-100  # Tolerance for numerical stability

        # Assert the validity of the inputs.
        assert 0 < self.linearity, "The linearity term must be positive."

    def forward(self, x):
        if self.forwardDirection != self.invertedActivation: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        return x + x / (1 + x.pow(2)) / self.linearity

    # TODO: unstable, diverges to infinity
    def inversePass(self, y):
        b, b2, b3 = self.linearity, self.linearity ** 2, self.linearity ** 3
        y2, y3, y4 = y.pow(2), y.pow(3), y.pow(4)

        # Compute components.
        term2 = 3 * b * (b + 1) - b2 * y2
        term1 = 2 * b3 * y3 + 18 * b3 * y - 9 * b2 * y
        N = term1 + torch.sqrt(torch.abs(4 * term2.pow(3) + term1.pow(2)))

        # Compute the cube root term
        signN = torch.nn.functional.hardtanh(N, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
        cube_root_term = signN * (N.abs() + self.tolerance).pow(1 / 3)

        # Compute x using the given equation
        x = (cube_root_term / (3 * (2 ** (1 / 3)) * b)) - ((2 ** (1 / 3)) * term2) / (3 * b * cube_root_term + self.tolerance) + y / 3

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

class reversibleActivationInterface(reversibleInterface):
    def __init__(self, activationFunctions):
        super(reversibleActivationInterface, self).__init__()
        self.activationFunctions = activationFunctions

    def forward(self, x): return self.activationFunctions(x)
