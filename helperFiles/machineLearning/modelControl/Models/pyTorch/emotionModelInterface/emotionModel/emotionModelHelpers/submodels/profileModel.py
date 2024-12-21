import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(profileModel, self).__init__()
        self.embeddedPhysiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments)
        self.profileStateLosses, self.compiledSignalEncoderLayerStatePath = None, None
        self.profileStatePath, self.embeddedProfileStatePath = None, None
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments

        # Initialize the physiological profile.
        self.resetProfileHolders()
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.physiologicalInitialization(self.embeddedPhysiologicalProfile)

    def resetProfileHolders(self):
        self.profileStateLosses, self.compiledSignalEncoderLayerStatePath = [], []
        self.profileStatePath, self.embeddedProfileStatePath = [], []

        # Get the model information.
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        # Pre-allocate each parameter.
        self.compiledSignalEncoderLayerStatePath.append(np.zeros(shape=(2*numSpecificEncoderLayers + numSharedEncoderLayers + 1, self.numExperiments, 1, self.encodedDimension)))
        self.embeddedProfileStatePath.append(torch.zeros(self.numExperiments, modelConstants.numEncodedWeights))
        self.profileStatePath.append(torch.zeros(self.numExperiments, self.encodedDimension))
        self.profileStateLosses.append(torch.zeros(self.numExperiments))

    def populateProfileState(self, batchInds, profileStateLoss, profileStatePath, compiledSignalEncoderLayerStatePath):
        self.embeddedProfileStatePath[-1][batchInds] = self.embeddedPhysiologicalProfile[batchInds].clone().detach().cpu().numpy()
        self.compiledSignalEncoderLayerStatePath[-1][:, batchInds] = compiledSignalEncoderLayerStatePath
        self.profileStateLosses[-1][batchInds] = profileStateLoss.clone().detach().cpu().numpy()
        self.profileStatePath[-1][batchInds] = profileStatePath.clone().detach().cpu().numpy()

    def getPhysiologicalProfile(self, batchInds):
        return self.embeddedPhysiologicalProfile.to(batchInds.device)[batchInds]
