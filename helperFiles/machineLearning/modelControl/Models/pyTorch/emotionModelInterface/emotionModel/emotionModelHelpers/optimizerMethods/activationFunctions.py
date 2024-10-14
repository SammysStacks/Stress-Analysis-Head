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
        infiniteBound = float(activationMethod.split('_')[2]) if '_' in activationMethod else 0.5
        activationFunction = reversibleLinearSoftSign(inversionPoint=inversionPoint, infiniteBound=infiniteBound)
    elif activationMethod == 'boundedS':
        activationFunction = boundedS()
    elif activationMethod == 'linearOscillation':
        activationFunction = linearOscillation()
    elif activationMethod == 'PReLU':
        activationFunction = nn.PReLU()
    elif activationMethod == 'selu':
        activationFunction = nn.SELU()
    elif activationMethod == 'gelu':
        activationFunction = nn.GELU()
    elif activationMethod == 'relu':
        activationFunction = nn.ReLU()
    elif activationMethod == 'sinh':
        activationFunction = sinh()
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


class reversiblePolynomial(reversibleInterface):
    def __init__(self, inversionPoint=2):
        super(reversiblePolynomial, self).__init__()
        self.squaredInversionPoint = inversionPoint ** 2
        self.inversionPoint = inversionPoint

    def forward(self, x):
        if self.forwardDirection:
            return self.forwardPass(x)
        else:
            return self.inversePass(x)

    def inversePass(self, x):
        return (x + x.pow(3) / self.squaredInversionPoint) / 2

    def forwardPass(self, y):
        # Solve the cubic equation terms.
        secondTerm = self.squaredInversionPoint * torch.sqrt(y.pow(2) + self.squaredInversionPoint / 27)
        firstTerm = self.squaredInversionPoint * y

        # Combine the cubic functions.
        secondRoot = firstTerm - secondTerm
        firstRoot = firstTerm + secondTerm

        # Invert the cubic function.
        x = torch.pow(firstRoot.abs(), 1 / 3) * firstRoot.sign() + torch.pow(secondRoot.abs(), 1 / 3) * secondRoot.sign()

        return x


class reversibleLinearSoftSign(reversibleInterface):
    def __init__(self, inversionPoint=2, infiniteBound=0.75):
        super(reversibleLinearSoftSign, self).__init__()
        self.inversionPoint = inversionPoint  # Corresponds to `r` in the equation
        self.infiniteBound = infiniteBound  # Corresponds to `a` in the equation

    def forward(self, x):
        if self.forwardDirection:
            return self.forwardPass(x)
        else:
            return self.inversePass(x)

    def forwardPass(self, x):
        # f(x) = ax + x / (1 + (a / (1 - a)) * (|x| / r))
        absX = x.abs()

        # Calculate the non-linearity term
        nonLinearityTerm = 1 + (self.infiniteBound / (1 - self.infiniteBound)) * (absX / self.inversionPoint)
        y = self.infiniteBound * x + x / nonLinearityTerm

        return y

    def inversePass(self, y):
        # Inverse function described in the problem
        absY = y.abs()

        # Calculate the non-linearity term
        squareRootTerm = ((self.infiniteBound ** 2 - 1) ** 2 * self.inversionPoint ** 2
                          - 2 * self.infiniteBound * self.inversionPoint * (self.infiniteBound - 1) ** 2 * absY
                          + self.infiniteBound ** 2 * y ** 2).sqrt()
        signDependentTerm = (squareRootTerm + (self.infiniteBound ** 2 - 1) * self.inversionPoint) / (2 * self.infiniteBound ** 2)
        signDependentTerm = signDependentTerm * y.sign()

        # Combine terms, applying sign(x) for the final output
        x = signDependentTerm + y / (2 * self.infiniteBound)

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

    def reverse(self, y):
        numeratorTerm = 1 - (y / self.nonLinearityRegion).pow(2)
        denominator = 2 * (1 + (y / self.nonLinearityRegion).pow(2))
        logTerm = torch.log(y)

        output = logTerm + numeratorTerm / denominator
        output = torch.exp(output)

        return output


# CHECK IF USING
class linearOscillation(nn.Module):
    def __init__(self, linearity=1, amplitude=1):
        super(linearOscillation, self).__init__()
        # NOTE MONOTONICITY: The linearity term must be greater or equal to the MAGNITUDE (ABS) of the amplitude to ensure the activation is monotonically increasing.
        self.linearity = linearity
        self.amplitude = amplitude

    def forward(self, x):
        # Ensure monotonicity.
        linearity = torch.clamp(self.linearity, min=abs(self.amplitude))

        return linearity * x + self.amplitude * torch.sin(x)


class boundedS(nn.Module):
    def __init__(self, boundedValue=1):
        super(boundedS, self).__init__()
        # Initialize coefficients with a starting value.
        self.coefficients = nn.Parameter(torch.tensor([0.01]))
        self.boundedValue = boundedValue

    def forward(self, x):
        # Update the coefficient clamp.
        a = self.coefficients[0].clamp(min=0.01, max=0.5)

        return (x / (1 + torch.pow(x, 2))) + a * x


class learnableBoundedS(nn.Module):
    def __init__(self):
        super(learnableBoundedS, self).__init__()
        # Initialize coefficients with a starting value.
        self.coefficients = nn.Parameter(torch.tensor([1.0000]))

    def forward(self, x):
        # Update the coefficient clamp.
        a = self.coefficients[0].clamp(min=1, max=100) + 25

        return a * x / (25 + torch.pow(x, 2))


class sinh(nn.Module):
    def __init__(self, clampCoeff=[0.5, 0.75]):
        super(sinh, self).__init__()
        # Initialize coefficients with a starting value.
        self.coefficients = nn.Parameter(torch.tensor(0.5))
        self.clampCoeff = clampCoeff

    def forward(self, x):
        # Update the coefficient clamp.
        coefficients = self.coefficients.clamp(min=self.clampCoeff[0], max=self.clampCoeff[1])

        return torch.sinh(coefficients * x)


class powerSeriesActivation(nn.Module):
    def __init__(self, numCoeffs=3, stabilityConstant=3.0, maxGrad=1, seriesType='full'):
        super(powerSeriesActivation, self).__init__()
        self.stabilityConstant = nn.Parameter(torch.tensor(stabilityConstant))
        self.coefficients = nn.Parameter(torch.ones(numCoeffs))
        self.seriesType = seriesType
        self.maxGrad = maxGrad

        # Register the hook with the coefficients
        self.stabilityConstant.register_hook(self.stabilityGradientHook)
        self.coefficients.register_hook(self.coeffGradientHook)

    def coeffGradientHook(self, grad):
        return grad.clamp(min=-self.maxGrad, max=self.maxGrad)

    def stabilityGradientHook(self, grad):
        # Clamp the gradients to be within the range [-self.stabilityConstant, self.stabilityConstant]
        return grad.clamp(min=-self.maxGrad, max=self.maxGrad)

    def forward(self, x):
        output = 0

        for coeffInd in range(len(self.coefficients)):
            functionPower = coeffInd + 1  # Skip the bias term.

            if self.seriesType == 'full':
                functionPower = functionPower  # Full series: f(x) = a_0*x + a_1*x^2 + ... + a_n*x^n
            elif self.seriesType == 'even':
                functionPower = 2 * functionPower  # Even series: f(x) = a_0*x^2 + a_1*x^4 + ... + a_n*x^(2n)
            elif self.seriesType == 'odd':
                functionPower = 2 * functionPower - 1  # Odd series: f(x) = a_0*x + a_1*x^3 + ... + a_n*x^(2n+1)
            else:
                raise NotImplementedError

            # Adjust the output.
            output += torch.exp(self.stabilityConstant) * torch.pow(x, functionPower) * self.coefficients[coeffInd]

        return output


class powerActivation(nn.Module):
    def __init__(self, initial_exponent=2.0, min_exponent=0.1, max_exponent=5.0):
        super(powerActivation, self).__init__()
        self.exponent = nn.Parameter(torch.tensor(initial_exponent))
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent

    def forward(self, x):
        # Apply constraints to the exponent
        constrained_exponent = torch.clamp(self.exponent, self.min_exponent, self.max_exponent)

        # Apply the power series activation
        return torch.pow(x, constrained_exponent)


class learnableTanhshrink(nn.Module):
    def __init__(self, initial_scale=1.0):
        super(learnableTanhshrink, self).__init__()
        # Initialize the learnable scale parameter
        self.nonLinearScale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        # Apply the tanh function to constrain the scale parameter between -1 and 1
        constrainedScale = torch.tanh(self.nonLinearScale)

        # Apply the Tanhshrink activation function with the learnable scale
        return x - constrainedScale * torch.tanh(x)


if __name__ == "__main__":
    # Test the activation functions
    data = torch.randn(2, 10, 100)

    # Perform the forward and inverse pass.
    activationClass = reversibleLinearSoftSign(inversionPoint=2, infiniteBound=0.5)
    _forwardData, _reconstructedData = activationClass.checkReconstruction(data, atol=1e-6)
