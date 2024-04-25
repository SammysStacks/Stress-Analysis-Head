# General
import torch
from torch import nn

# Import files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.nnHelpers.sharedModelWeights import sharedModelWeights
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.nnHelpers.specificModelWeights import specificModelWeights


class heatTherapyModel(nn.Module):
    def __init__(self, numTemperatures=1, numLosses=3, numTempBins=9, numLossBins=11):
        # General model parameters.
        super().__init__()
        # General model parameters.
        self.numTemperatures = numTemperatures  # The number of predicted temperatures.
        self.numTempBins = numTempBins   # The number of temperature bins.
        self.numLossBins = numLossBins   # The number of loss bins.
        self.numLosses = numLosses       # The number of losses.

        # Calculate the number of input shared features.
        self.numInputTempFeatures = self.numTemperatures + self.numLosses  # The number of input temperature features.
        self.numInputLossFeatures = self.numTemperatures*(self.numTempBins+1) + self.numLosses      # The number of input loss features.
        # Calculate the number of output shared features.
        self.numSharedTempFeatures = self.numTempBins*2     # The number of shared temperature features. Output dimension of the shared model.
        self.numSharedLossFeatures = self.numLosses*2       # The number of shared loss features. Output dimension of the shared model.

        # Shared model weights based on the population data.
        self.sharedModelWeights = sharedModelWeights(
            numSharedTempFeatures=self.numSharedTempFeatures,  # The number of shared temperature features.
            numSharedLossFeatures=self.numSharedLossFeatures,  # The number of shared loss features.
            numInputTempFeatures=self.numInputTempFeatures,    # The number of initial temperature features.
            numInputLossFeatures=self.numInputLossFeatures,    # The number of initial loss features.
        )

        # Specific model weights based on the individual data.
        self.specificModelWeights = specificModelWeights(
            numSharedTempFeatures=self.numSharedTempFeatures,   # The number of shared temperature features.
            numSharedLossFeatures=self.numSharedLossFeatures,   # The number of shared loss features.
            numTemperatures=self.numTemperatures,   # The number of predicted temperatures.
            numTempBins=self.numTempBins,   # The number of temperature bins.
            numLossBins=self.numLossBins,   # The number of loss bins.
            numLosses=self.numLosses,       # The number of losses.
        )

    def forward(self, initialPatientStates):
        batchSize, numInitialFeatures = initialPatientStates.size()
        # initialPatientStates dimensions: [batchSize, numInputFeatures=4].

        # Predict the next temperatures for the patient.
        finalTemperaturePredictions = self.predictNextTemperature(initialPatientStates)
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].

        # Update the patient's state.
        compiledTemperaturePredictions = finalTemperaturePredictions.transpose(0, 1).contiguous().view(batchSize, self.numTemperatures*self.numTempBins)
        nextPatientStates = torch.cat((compiledTemperaturePredictions, initialPatientStates), dim=1)
        # nextPatientStates dimensions: [batchSize, numInputLossFeatures].

        # Predict the expected loss of the patient.
        finalLossPredictions = self.predictNextState(nextPatientStates)
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].

        return finalTemperaturePredictions, finalLossPredictions

    def predictNextTemperature(self, patientStates):
        """ patientStates: The patient's current state. Dimensions: [batchSize, numInputFeatures=4]. """
        # Predict the next temperature of the patient.
        sharedTempFeatures = self.sharedModelWeights.sharedTempFeatureExtraction(patientStates)  # Extract the shared model features.
        finalTemperaturePredictions = self.specificModelWeights.predictNextTemperature(sharedTempFeatures)
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].

        return finalTemperaturePredictions

    def predictNextState(self, patientStates):
        # Predict the expected loss of the patient.
        sharedLossFeatures = self.sharedModelWeights.sharedLossFeatureExtraction(patientStates)  # Extract the shared model features.
        finalLossPredictions = self.specificModelWeights.predictNextLoss(sharedLossFeatures)
        # finalLossPrediction dimensions: [batchSize, numLosses].

        return finalLossPredictions
