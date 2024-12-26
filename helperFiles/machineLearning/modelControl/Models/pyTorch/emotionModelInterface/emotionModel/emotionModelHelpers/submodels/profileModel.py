import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, numSignals, encodedDimension):
        super(profileModel, self).__init__()
        self.embeddedPhysiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments)
        self.profileStateLosses, self.compiledSignalEncoderLayerStatePath = None, None
        self.profileStatePath, self.embeddedProfileStatePath = None, None
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.numSignals = numSignals

        # Initialize the physiological profile.
        self.resetProfileHolders(numProfileShots=int(modelConstants.useInitialLoss))
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.physiologicalInitialization(self.embeddedPhysiologicalProfile)

    def resetProfileHolders(self, numProfileShots):
        # Get the model information.
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        # Pre-allocate each parameter.
        self.compiledSignalEncoderLayerStatePath = np.zeros(shape=(numProfileShots + 1, 2*numSpecificEncoderLayers + numSharedEncoderLayers + 1, self.numExperiments, 1, self.encodedDimension))
        self.embeddedProfileStatePath = np.zeros(shape=(numProfileShots + 1, self.numExperiments, modelConstants.numEncodedWeights))
        self.profileStatePath = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.encodedDimension))
        self.profileStateLosses = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals))

    def populateProfileState(self, profileEpoch, batchInds, profileStateLoss, profileStatePath, compiledSignalEncoderLayerStatePath):
        if isinstance(batchInds, torch.Tensor): batchInds = batchInds.detach().cpu().numpy()
        self.embeddedProfileStatePath[profileEpoch][batchInds] = self.embeddedPhysiologicalProfile[batchInds].clone().detach().cpu().numpy()
        self.compiledSignalEncoderLayerStatePath[profileEpoch][:, batchInds] = compiledSignalEncoderLayerStatePath
        self.profileStateLosses[profileEpoch][batchInds] = profileStateLoss.clone().detach().cpu().numpy()
        self.profileStatePath[profileEpoch][batchInds] = profileStatePath.clone().detach().cpu().numpy()

    def getPhysiologicalProfile(self, batchInds):
        return self.embeddedPhysiologicalProfile.to(batchInds.device)[batchInds]
