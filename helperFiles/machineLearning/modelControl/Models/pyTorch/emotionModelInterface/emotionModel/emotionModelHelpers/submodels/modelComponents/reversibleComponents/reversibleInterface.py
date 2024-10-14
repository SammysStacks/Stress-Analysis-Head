import torch
from matplotlib import pyplot as plt
from torch import nn


class reversibleInterface(nn.Module):
    forwardDirection = True

    @classmethod
    def changeDirections(cls, forwardDirection):
        cls.forwardDirection = forwardDirection  # Modify class attribute

    @staticmethod
    def getStabilityTerm(kernelSize, scalingFactor, device):
        return torch.eye(kernelSize, device=device)*scalingFactor

    def checkReconstruction(self, inputData, atol=1e-8):
        # Forward direction.
        reversibleInterface.changeDirections(True)
        forwardData = self.forward(inputData)

        # Backward direction.
        reversibleInterface.changeDirections(False)
        reconstructedData = self.forward(forwardData)

        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=atol): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(inputData[0][0].detach().numpy(), 'k', linewidth=2, label='Initial Signal')
        plt.plot(reconstructedData[0][0].detach().numpy(), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
        plt.plot(forwardData[0][0].detach().numpy(), 'tab:blue', linewidth=1, label='Latent Signal', alpha=0.5)
        plt.legend()
        plt.show()

        plt.plot((inputData - reconstructedData)[0][0].detach().numpy(), 'k', linewidth=2, label='Signal Error')
        plt.legend()
        plt.show()

        return forwardData, reconstructedData
