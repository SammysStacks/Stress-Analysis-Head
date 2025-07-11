import math

import numpy as np
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
    elif activationMethod == 'PReLU':
        activationFunction = nn.PReLU()
    elif activationMethod == 'selu':
        activationFunction = nn.SELU()
    elif activationMethod == 'gelu':
        activationFunction = nn.GELU()
    elif activationMethod == 'relu':
        activationFunction = nn.ReLU()
    elif activationMethod == 'SoftSign':
        activationFunction = nn.Softsign()
    else: raise ValueError("Activation type must be in ['Tanhshrink', 'none', 'boundedExp', 'reversibleLinearSoftSign', 'boundedS', 'PReLU', 'selu', 'gelu', 'relu']")

    return activationFunction


class reversibleLinearSoftSign(reversibleInterface):
    def __init__(self):
        super(reversibleLinearSoftSign, self).__init__()
        self.infiniteBoundParam = nn.Parameter(torch.zeros(1))  # The infinite bound controller.
        self.convergencePointParam = nn.Parameter(torch.zeros(1))  # The infinite bound controller.
        self.infiniteBound = torch.zeros(1)  # The infinite bound parameter.
        self.nonLinearCoefficient = torch.zeros(1)  # The nonLinearCoefficient parameter.
        self.tolerance = 1e-25  # Tolerance for numerical stability

    def getActivationParams(self):
        infiniteBound = 0.5 + 0.4*torch.tanh(self.infiniteBoundParam)  # Convert the infinite bound to a sigmoid value. Maybe 0.5 +/- 0.25
        convergentPoint = 1 + 0.5*torch.tanh(self.convergencePointParam)  # Convert the infinite bound to a sigmoid value. Maybe 1 +/-0.5
        nonLinearCoefficient = (1 + convergentPoint) * (1 - infiniteBound)  # 1/r

        # Assert the validity of the parameters.
        assert infiniteBound != 1, "The infinite bound must not be equal to 1, as it would lead to a division by zero in the inverse pass."

        # Return the parameters.
        return infiniteBound, nonLinearCoefficient, convergentPoint

    def forward(self, x, linearModel, forwardFirst=True):
        # Set the parameters for the forward and inverse passes.
        self.infiniteBound, self.nonLinearCoefficient, _ = self.getActivationParams()

        # forwardPass: Increase the signal below inversion point; decrease above.
        x = self.forwardPass(x) if forwardFirst else self.inversePass(x)
        x = linearModel(x)  # Rotate the signal through the linear model.

        # inversePass: Decrease the signal below inversion point; increase above.
        x = self.inversePass(x) if forwardFirst else self.forwardPass(x)

        return x

    def forwardPass(self, x):
        # Increase the signal below inversion point; decrease above.
        return self.infiniteBound*x + self.nonLinearCoefficient * x / (1 + x.abs())  # f(x) = a*x + x / (1 + |x|) / r

    def inversePass(self, y):
        # Prepare the terms for the inverse pass.
        signY = torch.nn.functional.hardtanh(y, min_val=-self.tolerance, max_val=self.tolerance) / self.tolerance
        r, a = 1/self.nonLinearCoefficient, self.infiniteBound  # The nonLinearCoefficient and infinite-bound terms

        # Base case: infiniteBound=0.
        if a == 0: return y*r / (1 - signY*y*r)  # Poor numerical stability on reconstruction!

        # Decrease the signal below inversion point; increase above.
        sqrtTerm = ((r*a)**2 + 2*a*r*(1 + signY*y*r) + (r*y - signY).pow(2)) / (r*a)**2
        x = signY*(sqrtTerm.sqrt() - 1)/2 - signY / (2*a*r) + y / (2*a)

        return x

    def getActivationCurve(self, x_min=-2, x_max=2, num_points=200):
        # Turn off gradient tracking for plotting
        with torch.no_grad():
            x_vals = torch.linspace(x_min, x_max, num_points, device=self.infiniteBound.device)
            y_vals = self.forwardPass(x_vals)

        # Convert to NumPy for plotting
        return x_vals.detach().cpu().numpy().astype(np.float16), y_vals.detach().cpu().numpy().astype(np.float16)


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
