import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, numSignals, encodedDimension):
        super(profileModel, self).__init__()
        self.embeddedHealthProfiles = self.getInitialPhysiologicalProfile(numExperiments)
        self.retrainingProfileLosses, self.generatingBiometricSignals = None, None
        self.retrainingHealthProfilePath = None
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.numSignals = numSignals

        # Initialize the health profile.
        self.resetProfileHolders(numProfileShots=int(modelConstants.useInitialLoss))
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.healthInitialization(self.embeddedHealthProfiles)

    def resetProfileHolders(self, numProfileShots):
        # Pre-allocate each parameter.
        self.generatingBiometricSignals = np.zeros(shape=(numProfileShots + 1, 1, self.numExperiments, self.numSignals, self.encodedDimension))  # Dim: numProfileShots, numLayers, numExperiments, numSignals, encodedDimension
        self.retrainingHealthProfilePath = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.encodedDimension))
        self.retrainingProfileLosses = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals))

    def populateProfileState(self, profileEpoch, batchInds, profileStateLoss, generatingBiometricSignals, healthProfile):
        if isinstance(batchInds, torch.Tensor): batchInds = batchInds.detach().cpu().numpy()
        self.retrainingHealthProfilePath[profileEpoch][batchInds] = healthProfile.clone().detach().cpu().numpy()
        self.retrainingProfileLosses[profileEpoch][batchInds] = profileStateLoss.clone().detach().cpu().numpy()
        self.generatingBiometricSignals[profileEpoch][:, batchInds] = generatingBiometricSignals[:, :, :, :].copy()

    def getHealthEmbedding(self, batchInds):
        return self.embeddedHealthProfiles.to(batchInds.device)[batchInds]
