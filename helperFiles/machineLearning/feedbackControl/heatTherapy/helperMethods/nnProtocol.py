# General
import torch
from torch import nn, optim

# Import files.
from .generalProtocol import generalProtocol
from .nnHelpers.heatTherapyModel import heatTherapyModel


class nnProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, modelName, onlineTraining=False):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # General model parameters.
        self.onlineTraining = onlineTraining  # Whether to train the model live.
        self.modelName = modelName  # The model's unique identifier.
        self.optimizer = None       # The optimizer for the model.

        self.predictedLosses = []

        # Model parameters.
        self.model = heatTherapyModel(numTemperatures=1, numLosses=3, numTempBins=9, numLossBins=11)  # The model for the therapy.

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = self.userStatePath[-1]
        currentUserTemp, currentUserLoss = currentUserState

        # Update the temperatures visited.
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)

        # TODO: Get the new user temperature and predicted loss.
        finalTemperaturePredictions, finalLossPredictions = self.model(currentUserState)
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].

        newUserTemp = finalTemperaturePredictions.argmax(dim=2)  # Not differentiable
        # newUserTemp dimensions: [numTemperatures, batchSize].

        self.predictedLosses.append(finalLossPredictions)

        return newUserTemp, (None,)

    def updateWeights(self, actualLoss):
        # TODO: Backpropagate the loss.
        pass

    # ------------------------ Machine Learning ------------------------ #

    def setupModelHelpers(self):
        # Define the optimizer.
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=1e-3)
