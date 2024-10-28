import math

import torch
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


def getActivationMethod(activationMethod):
    if activationMethod == 'Tanhshrink':
        activationFunction = nn.Tanhshrink()
    elif activationMethod == 'none':
        activationFunction = nn.Identity()
    elif activationMethod.startswith('boundedExp'):
        nonLinearityRegion = int(activationMethod.split('_')[2]) if '_' in activationMethod else 2
        topExponent = int(activationMethod.split('_')[1]) if '_' in activationMethod else 0
        activationFunction = boundedExp(decayConstant=topExponent, nonLinearityRegion=nonLinearityRegion)
    elif activationMethod.startswith('reversibleLinearSoftSign'):
        inversionPoint = float(activationMethod.split('_')[1]) if '_' in activationMethod else 2
        infiniteBound = float(activationMethod.split('_')[2]) if '_' in activationMethod else 1
        activationFunction = reversibleLinearSoftSign(inversionPoint=inversionPoint, infiniteBound=infiniteBound)
    elif activationMethod.startswith('nonLinearMultiplication'):
        invertedActivation = activationMethod.split('_')[1] == "True"
        activationFunction = nonLinearMultiplication(invertedActivation=invertedActivation)
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
    def __init__(self, inversionPoint=1, infiniteBound=0.75):
        super(reversibleLinearSoftSign, self).__init__()
        self.inversionPoint = inversionPoint  # Corresponds to `r` in the equation
        self.infiniteBound = infiniteBound  # Corresponds to `a` in the equation

        # Assert the validity of the inputs.
        assert 0 < self.infiniteBound <= 1, "The magnitude of the inf bound has a domain of (0, 1] to ensure a stable convergence."
        assert 0 < self.inversionPoint, "The inversion point must be positive to ensure a stable convergence."

    def forward(self, x):
        if self.forwardDirection: return self.forwardPass(x)
        else: return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (|x| / r))
        absX = x.abs()

        # Calculate the non-linearity term
        if self.infiniteBound != 1: nonLinearityTerm = 1 + (self.infiniteBound / (1 - self.infiniteBound)) * (absX / self.inversionPoint)
        else: nonLinearityTerm = 1 + (absX / self.inversionPoint)

        # Calculate the output.
        y = self.infiniteBound * x + x / nonLinearityTerm

        return y

    def inversePass(self, y):
        # Inverse function described in the problem
        absY = y.abs()

        if self.infiniteBound != 1:
            # Calculate the non-linearity term
            squareRootTerm = ((self.infiniteBound ** 2 - 1) ** 2 * self.inversionPoint ** 2
                              - 2 * self.infiniteBound * self.inversionPoint * (self.infiniteBound - 1) ** 2 * absY
                              + self.infiniteBound ** 2 * y ** 2).sqrt()
            signDependentTerm = (squareRootTerm + (self.infiniteBound ** 2 - 1) * self.inversionPoint) / (2 * self.infiniteBound ** 2)
            signDependentTerm = signDependentTerm * y.sign()

            # Combine terms, applying sign(x) for the final output
            x = signDependentTerm + y / (2 * self.infiniteBound)
        else:
            signY = torch.nn.Tanh()(y*1e5)
            # Calculate the non-linearity term
            x = signY*((4*self.inversionPoint**2 + y.pow(2)).sqrt() - 2*self.inversionPoint)/2 + y/2

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


class signWrap(reversibleInterface):
    def __init__(self, period=1):
        super(reversibleInterface, self).__init__()
        self.period = torch.as_tensor(period)  # Assign r as nonLinearity
        assert period != 0, "The period parameter must be non-zero."

    def forward(self, x):
        wrappedData = self.arcsin(self.sin(x))

        if self.forwardDirection: return self.forwardPass(wrappedData)
        else: return self.inversePass(wrappedData)

    def forwardPass(self, x):
        return self.sin(x)

    def inversePass(self, y):
        return self.arcsin(y)

    def sin(self, x):
        return self.period*(x*torch.pi/self.period).sin()

    def arcsin(self, x):
        return (self.period/torch.pi) * (x/self.period).asin()


class nonLinearMultiplication(reversibleInterface):
    def __init__(self, invertedActivation):
        super(nonLinearMultiplication, self).__init__()
        self.invertedActivation = invertedActivation  # Whether the non-linearity term is inverted
        self.sequenceLength = None  # The length of the input signal
        self.amplitude = 0.5

        # Create a learnable parameter, initialized to the given initial value
        self.learnablePhaseShift = nn.Parameter(torch.as_tensor(torch.pi))  # The phase shift of the non-linearity term.
        self.learnableFrequency = nn.Parameter(torch.as_tensor(0.5))  # The frequency of the non-linearity term.

        # Register hooks for each parameter in the list
        self.learnablePhaseShift.register_hook(self.scaleGradients)
        self.learnableFrequency.register_hook(self.scaleGradients)

    @staticmethod
    def scaleGradients(grad):
        return grad * 0.1

    def forward(self, x):
        # Check if the non-linearity term has been calculated.
        if self.sequenceLength is None: self.sequenceLength = x.size(-1)
        assert x.size(-1) == self.sequenceLength, "The sequence length of the input data must match the sequence length of the non-linearity term."

        # Get the non-linearity term.
        nonLinearityTerm = self.getNonLinearity(device=x.device)

        if self.forwardDirection != self.invertedActivation: return self.inversePass(x, nonLinearityTerm)
        else: return self.forwardPass(x, nonLinearityTerm)

    @staticmethod
    def forwardPass(x, nonLinearityTerm):
        return x * nonLinearityTerm

    @staticmethod
    def inversePass(y, nonLinearityTerm):
        return y / nonLinearityTerm

    def getNonLinearity(self, device):
        positions = torch.arange(start=0, end=self.sequenceLength, step=1, dtype=torch.float32, device=device)

        return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2


if __name__ == "__main__":
    # Test the activation functions
    data = torch.randn(2, 10, 100, dtype=torch.float64)
    data = data - data.min()
    data = data / data.max()
    data = 2 * data - 1

    # Perform the forward and inverse pass.
    activationClass = nonLinearMultiplication(invertedActivation=False)
    _forwardData, _reconstructedData = activationClass.checkReconstruction(data, atol=1e-6, numLayers=10)
