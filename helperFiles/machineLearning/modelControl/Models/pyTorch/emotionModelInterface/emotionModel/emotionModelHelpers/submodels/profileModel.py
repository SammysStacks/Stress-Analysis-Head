import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, numSignals, encodedDimension):
        super(profileModel, self).__init__()
        self.embeddedHealthProfiles = self.getInitialPhysiologicalProfile(numExperiments)
        self.retrainingProfileLosses, self.generatingBiometricSignals = None, None
        self.compiledLayerStates, self.compiledLayerStateInd = None, 0
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
        self.generatingBiometricSignals = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals, self.encodedDimension))  # Dim: numProfileShots, numLayers, numExperiments, numSignals, encodedDimension
        self.retrainingHealthProfilePath = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.encodedDimension))
        self.retrainingProfileLosses = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals))

    def resetModelStates(self, metaLearningData):
        # Pre-allocate each parameter.
        numSpecificEncoderLayers, numSharedEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers'], modelConstants.userInputParams['numSharedEncoderLayers']
        self.compiledLayerStates = np.zeros(shape=(numSpecificEncoderLayers + numSharedEncoderLayers + 1, metaLearningData.shape[0], metaLearningData.shape[1], metaLearningData.shape[2]))
        self.compiledLayerStateInd = 0

        # Add the initial state.
        self.addModelState(metaLearningData)

    def addModelState(self, metaLearningData):
        self.compiledLayerStates[self.compiledLayerStateInd] = metaLearningData.clone().detach().cpu().numpy().reshape((self.compiledLayerStates.shape[1], self.compiledLayerStates.shape[2], self.compiledLayerStates.shape[3]))
        self.compiledLayerStateInd += 1

    def populateProfileState(self, profileEpoch, batchInds, profileStateLoss, resampledSignalData, healthProfile):
        if isinstance(batchInds, torch.Tensor): batchInds = batchInds.detach().cpu().numpy()
        self.retrainingProfileLosses[profileEpoch][batchInds] = profileStateLoss.clone().detach().cpu().numpy()
        self.retrainingHealthProfilePath[profileEpoch][batchInds] = healthProfile.clone().detach().cpu().numpy()
        self.generatingBiometricSignals[profileEpoch][batchInds] = resampledSignalData.clone().detach().cpu().numpy()

    def getHealthEmbedding(self, batchInds):
        return self.embeddedHealthProfiles.to(batchInds.device)[batchInds]
