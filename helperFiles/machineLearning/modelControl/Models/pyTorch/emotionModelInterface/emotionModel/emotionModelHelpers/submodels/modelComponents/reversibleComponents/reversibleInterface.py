import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class reversibleInterface(nn.Module):
    forwardDirection = True

    def __init__(self):
        super(reversibleInterface, self).__init__()

    @classmethod
    def changeDirections(cls, forwardDirection):
        cls.forwardDirection = forwardDirection  # Modify class attribute

    def checkReconstruction(self, inputData, atol=1e-8, numLayers=10, plotResults=True):
        # Initialize the forward data.
        reversibleInterface.changeDirections(False)
        forwardData = inputData.clone()
        t1 = time.time()

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
        t2 = time.time(); print(f"\nTime taken for {numLayers} layers: {t2 - t1}")
        if plotResults: self.plotReconstruction(inputData, forwardData, reconstructedData, atol=atol, numPlots=2)

        return forwardData, reconstructedData

    @staticmethod
    def plotReconstruction(inputData, forwardData, reconstructedData, atol=1e-8, numPlots=1):
        # Compare the original and reconstructed inputData
        if torch.allclose(inputData, reconstructedData, atol=atol): print("Successfully reconstructed the original inputData!")
        else: print("Reconstruction failed. There is a discrepancy between the original and reconstructed inputData.")

        for signalInd in range(min(numPlots, inputData.size(1))):
            # Optionally, plot the original and reconstructed signals for visual comparison
            plt.plot(inputData[0][signalInd].detach().numpy().astype(np.float16), 'k', linewidth=2, label='Initial Signal')
            plt.plot(reconstructedData[0][signalInd].detach().numpy().astype(np.float16), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
            plt.plot(forwardData[0][signalInd].detach().numpy().astype(np.float16), 'o', color='tab:blue', linewidth=1, label='Latent Signal', alpha=0.5)
            plt.legend()
            plt.show()

            plt.plot((inputData - reconstructedData)[0][signalInd].detach().numpy().astype(np.float16), 'ko', linewidth=2, label='Signal Error')
            plt.legend()
            plt.show()

            plt.plot((inputData - forwardData)[0][signalInd].detach().numpy().astype(np.float16), 'ko', linewidth=2, label='Signal Change')
            plt.legend()
            plt.show()

        return forwardData, reconstructedData
