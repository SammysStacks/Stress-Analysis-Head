import time

import torch
from matplotlib import pyplot as plt
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.weightInitialization import weightInitialization


class reversibleInterface(nn.Module):
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

        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=atol): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        for signalInd in range(min(2, inputData.size(1))):
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
