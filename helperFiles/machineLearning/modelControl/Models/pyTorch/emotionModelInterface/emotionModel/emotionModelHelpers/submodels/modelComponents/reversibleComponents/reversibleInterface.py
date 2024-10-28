import time

import torch
from matplotlib import pyplot as plt
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.weightInitialization import weightInitialization


class reversibleInterface(nn.Module):
    switchActivationDirection = False
    forwardDirection = True

    def __init__(self):
        super(reversibleInterface, self).__init__()
        self.weightInitialization = weightInitialization()

    @classmethod
    def changeDirections(cls, forwardDirection):
        cls.forwardDirection = forwardDirection  # Modify class attribute

    @staticmethod
    def getStabilityTerm(kernelSize, scalingFactor, device):
        return torch.eye(kernelSize, device=device)*scalingFactor

    def checkDualReconstruction(self, x1, x2, atol=1e-8, numLayers=10):
        t1 = time.time()
        # Initialize the forward data.
        reversibleInterface.changeDirections(False)
        f1, f2 = x1.clone().double(), x2.clone().double()

        # Perform the forward passes.
        for layerInd in range(numLayers):
            f1, f2 = self.forward(f1, f2)

        # Initialize the backward data.
        reversibleInterface.changeDirections(True)
        r1, r2 = f1.clone(), f2.clone()

        # Perform the backward passes.
        for layerInd in range(numLayers):
            r1, r2 = self.forward(r1, r2)

        # Calculate the time taken for the forward and backward passes.
        t2 = time.time(); print(f"Time taken for {numLayers} layers: {t2 - t1}")
        self.plotReconstruction(x1, f1, r1, atol=atol, numPlots=1)
        self.plotReconstruction(x2, f2, r2, atol=atol, numPlots=1)

        return f1, f2, r1, r2

    def checkReconstruction(self, inputData, atol=1e-8, numLayers=10):
        t1 = time.time()
        # Initialize the forward data.
        reversibleInterface.changeDirections(False)
        forwardData = inputData.clone().double()

        # Perform the forward passes.
        for layerInd in range(numLayers):
            forwardData = self.forward(forwardData)

        # Initialize the backward data.
        reversibleInterface.changeDirections(True)
        reconstructedData = forwardData.clone()

        # Perform the backward passes.
        for layerInd in range(numLayers):
            reconstructedData = self.forward(reconstructedData)

        # Calculate the time taken for the forward and backward passes.
        t2 = time.time(); print(f"Time taken for {numLayers} layers: {t2 - t1}")
        self.plotReconstruction(inputData, forwardData, reconstructedData, atol=atol, numPlots=2)

        return forwardData, reconstructedData

    @staticmethod
    def plotReconstruction(inputData, forwardData, reconstructedData, atol=1e-8, numPlots=1):
        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=atol): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        for signalInd in range(min(numPlots, inputData.size(1))):
            # Optionally, plot the original and reconstructed signals for visual comparison
            plt.plot(inputData[0][signalInd].detach().numpy(), 'k', linewidth=2, label='Initial Signal')
            plt.plot(reconstructedData[0][signalInd].detach().numpy(), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
            plt.plot(forwardData[0][signalInd].detach().numpy(), 'o', color='tab:blue', linewidth=1, label='Latent Signal', alpha=0.5)
            plt.legend()
            plt.show()

            plt.plot((inputData - reconstructedData)[0][signalInd].detach().numpy(), 'k', linewidth=2, label='Signal Error')
            plt.legend()
            plt.show()

            plt.plot((inputData - forwardData)[0][signalInd].detach().numpy(), 'k', linewidth=2, label='Signal Change')
            plt.legend()
            plt.show()

        return forwardData, reconstructedData
