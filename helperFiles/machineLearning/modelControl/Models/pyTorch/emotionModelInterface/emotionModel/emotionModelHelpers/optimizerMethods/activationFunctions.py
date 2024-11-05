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
    elif activationMethod.startswith('reversibleActivation'):
        invertedActivation = activationMethod.split('_')[1] == "True"
        activationFunction = reversibleActivation(invertedActivation=invertedActivation)
    elif activationMethod == 'PReLU':
        activationFunction = nn.PReLU()
    elif activationMethod == 'selu':
        activationFunction = nn.SELU()
    elif activationMethod == 'gelu':
        activationFunction = nn.GELU()
    elif activationMethod == 'relu':
        activationFunction = nn.ReLU()
    else:
        raise ValueError("Activation type must be in ['Tanhshrink', 'none', 'boundedExp', 'boundedS' 'PReLU', 'selu', 'gelu', 'relu', 'sinh']")

    return activationFunction


class switchActivation(nn.Module):
    def __init__(self, activationFunction, switchState=True):
        super(switchActivation, self).__init__()
        self.activationFunction = activationFunction
        self.switchState = switchState

    def forward(self, x):
        if self.switchState:
            return self.activationFunction(x)
        else:
            return x


class reversibleLinearSoftSign(reversibleInterface):
    def __init__(self, invertedActivation=False, inversionPoint=2, infiniteBound=1):
        super(reversibleLinearSoftSign, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.inversionPoint = inversionPoint  # Corresponds to `r` in the equation
        self.infiniteBound = infiniteBound  # Corresponds to `a` in the equation

        # Assert the validity of the inputs.
        assert 0 < self.infiniteBound <= 1, "The magnitude of the inf bound has a domain of (0, 1] to ensure a stable convergence."
        assert 0 < self.inversionPoint, "The inversion point must be positive to ensure a stable convergence."

    def forward(self, x):
        if self.forwardDirection != self.invertedActivation: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (|x| / r))
        absX = x.abs()  # Not ideal for backpropagation, but necessary for the non-linearity term

        # Calculate the non-linearity term
        if self.infiniteBound != 1: nonLinearityTerm = 1 + (self.infiniteBound / (1 - self.infiniteBound)) * (absX / self.inversionPoint)
        else: nonLinearityTerm = 1 + (absX / self.inversionPoint)

        # Calculate the output.
        y = self.infiniteBound * x + x / nonLinearityTerm

        return y

    def inversePass(self, y):
        signY = torch.nn.Tanh()(y * 1e3)

        if self.infiniteBound != 1:
            absY = y*signY
            # Calculate the non-linearity term
            squareRootTerm = ((self.infiniteBound ** 2 - 1) ** 2 * self.inversionPoint ** 2
                              - 2 * self.infiniteBound * self.inversionPoint * (self.infiniteBound - 1) ** 2 * absY
                              + self.infiniteBound ** 2 * y ** 2).sqrt()
            signDependentTerm = (squareRootTerm + (self.infiniteBound ** 2 - 1) * self.inversionPoint) / (2 * self.infiniteBound ** 2)
            signDependentTerm = signDependentTerm * y.sign()

            # Combine terms, applying sign(x) for the final output
            x = signDependentTerm + y / (2 * self.infiniteBound)
        else:
            # Calculate the non-linearity term
            x = signY*((4*self.inversionPoint**2 + y.pow(2)).sqrt() - 2*self.inversionPoint)/2 + y/2

        return x

class reversibleActivation(reversibleInterface):
    def __init__(self, invertedActivation=False, inversionPoint=0.1, infiniteBound=1):
        super(reversibleActivation, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.inversionPoint = inversionPoint  # Corresponds to `r` in the equation
        self.infiniteBound = infiniteBound  # Corresponds to `a` in the equation
        self.tolerance = 1e-25

        # Assert the validity of the inputs.
        assert 0 < self.infiniteBound <= 1, "The magnitude of the inf bound has a domain of (0, 1] to ensure a stable convergence."
        assert 0 < self.inversionPoint, "The inversion point must be positive to ensure a stable convergence."
        assert infiniteBound == 1, "The infinite bound must be 1 for the activation to be stable."

    def forward(self, x):
        if self.forwardDirection != self.invertedActivation: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (xx / rr))
        return x + x / (1 + (x / self.inversionPoint).pow(2))

    def inversePass(self, y):
        r = self.inversionPoint

        # Compute the square root term
        sqrt_term = torch.sqrt(96 * r ** 6 - 39 * r ** 4 * y ** 2 + 12 * r ** 2 * y ** 4)
        # torch.sqrt(96 r^6 - 39 r^4 y^2 + 12 r^2 y^4)

        # Compute the numerator inside the cube root
        N = 9 * (r ** 2) * y + 3 * sqrt_term + 2 * y ** 3
        signN = torch.nn.functional.hardtanh(N, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
        # N = 9 r^2 y + 3 sqrt_term + 2 y^3

        # Compute the cube root term, handling negative values
        cube_root_term = signN * torch.abs(N).pow(1/3)

        # Compute numerator1 and numerator2, adding epsilon to avoid division by zero
        numerator2 = (2 ** (4/3)) * (6 * r ** 2 - y ** 2) / cube_root_term
        numerator1 = (2 ** (2/3)) * cube_root_term

        # Compute x
        x = (numerator1 - numerator2 + 2 * y) / 6

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


class boundedS(reversibleInterface):
    def __init__(self, nonLinearity):
        super(boundedS, self).__init__()
        self.nonLinearity = torch.as_tensor(nonLinearity)  # Assign r as nonLinearity
        assert 1 <= self.nonLinearity, "The non-linearity parameter must be greater than or equal to 1."

    def forward(self, x):
        if self.forwardDirection: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # Update the coefficient clamp.
        r = self.nonLinearity

        return x*(1 + 1/(1 + r*x.pow(2)))

    def inversePass(self, y):
        r = self.nonLinearity
        ySign = y.sign()
        yAbs = y.abs()

        # Compute terms step by step for numerical stability
        # Avoid any computation under the square root from becoming negative due to rounding or instability
        inner_term_sqrt = torch.clamp(4 * r**2 * yAbs**4 - 13 * r * yAbs**2 + 32, min=1e-200)
        term1_inner_sqrt = math.sqrt(3) * torch.sqrt(r**3 * inner_term_sqrt)

        term1_numerator = torch.clamp(2 * r**3 * yAbs**3 + 9 * r**2 * yAbs + term1_inner_sqrt, min=1e-200)
        term1 = torch.pow(term1_numerator, 1 / 3)

        term2 = 2 * (r * yAbs**2 - 6)

        term3_inner_sqrt = math.sqrt(3) * torch.sqrt(r**3 * inner_term_sqrt)
        term3_numerator = torch.clamp(r**3 * yAbs**3 + (9 * r**2 * yAbs) / 2 + (3 / 2) * term3_inner_sqrt, min=1e-200)
        term3 = torch.pow(term3_numerator, 1 / 3)

        # Final expression for x, using clamp to prevent overflow/underflow where necessary
        x = (1 / 6) * ((2**(2 / 3) * term1) / r + term2 / term3 + 2 * yAbs)

        return x*ySign


if __name__ == "__main__":
    # Test the activation functions
    data = torch.randn(2, 10, 100, dtype=torch.float64)
    data = data - data.min()
    data = data / data.max()
    data = 4 * data - 2

    # Perform the forward and inverse pass.
    activationClass = reversibleActivation(invertedActivation=True)
    _forwardData, _reconstructedData = activationClass.checkReconstruction(data, atol=1e-6, numLayers=2)
