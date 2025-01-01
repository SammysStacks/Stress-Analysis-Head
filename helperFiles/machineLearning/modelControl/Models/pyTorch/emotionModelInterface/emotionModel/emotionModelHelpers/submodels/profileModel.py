import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, numSignals, encodedDimension):
        super(profileModel, self).__init__()
        self.embeddedHealthProfiles = self.getInitialPhysiologicalProfile(numExperiments)
        self.retrainingProfileLosses, self.signalEncoderLayerTransforms = None, None
        self.retrainingEmbeddedProfilePath = None
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.numSignals = numSignals

        # Initialize the health profile.
        self.resetProfileHolders(numProfileShots=int(modelConstants.useInitialLoss))
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.healthInitialization(self.embeddedHealthProfiles)

    def resetProfileHolders(self, numProfileShots):
        # Get the model information.
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        # Pre-allocate each parameter.
        self.signalEncoderLayerTransforms = np.zeros(shape=(numProfileShots + 1, 2 * numSpecificEncoderLayers + numSharedEncoderLayers + 1, self.numExperiments, 1, self.encodedDimension))
        self.retrainingEmbeddedProfilePath = np.zeros(shape=(numProfileShots + 1, self.numExperiments, modelConstants.numEncodedWeights))
        self.retrainingProfileLosses = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals))

    def populateProfileState(self, profileEpoch, batchInds, profileStateLoss, signalEncoderLayerTransforms):
        if isinstance(batchInds, torch.Tensor): batchInds = batchInds.detach().cpu().numpy()
        self.retrainingEmbeddedProfilePath[profileEpoch][batchInds] = self.embeddedHealthProfiles[batchInds].clone().detach().cpu().numpy()
        self.retrainingProfileLosses[profileEpoch][batchInds] = profileStateLoss.clone().detach().cpu().numpy()
        self.signalEncoderLayerTransforms[profileEpoch][:, batchInds] = signalEncoderLayerTransforms

    def getHealthEmbedding(self, batchInds):
        return self.embeddedHealthProfiles.to(batchInds.device)[batchInds]
