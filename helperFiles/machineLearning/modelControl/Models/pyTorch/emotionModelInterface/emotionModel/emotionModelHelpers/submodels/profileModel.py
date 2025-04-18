import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, numSignals, encodedDimension):
        super(profileModel, self).__init__()
        self.fourierDimension = modelConstants.userInputParams['profileDimension'] // 2 + 1
        self.retrainingProfileLosses, self.generatingBiometricSignals = None, None
        self.compiledLayerStates, self.compiledLayerStateInd = None, 0
        self.encodedDimension = encodedDimension
        self.retrainingHealthProfilePath = None
        self.numExperiments = numExperiments
        self.numSignals = numSignals

        # Initialize the health profile.
        self.imaginaryProfileCoefficients = self.getInitialPhysiologicalProfile(numExperiments, self.fourierDimension)
        self.realProfileCoefficients = self.getInitialPhysiologicalProfile(numExperiments, self.fourierDimension)
        self.resetProfileHolders(numProfileShots=modelConstants.userInputParams['numProfileShots'] + 1)
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.healthInitialization(self.imaginaryProfileCoefficients)
        self.healthInitialization(self.realProfileCoefficients)

    def resetProfileHolders(self, numProfileShots):
        # Pre-allocate each parameter.
        self.generatingBiometricSignals = np.zeros(shape=(numProfileShots + 1, 1, self.numSignals, self.encodedDimension), dtype=np.float16)  # Dim: numProfileShots, numExperiments, numSignals, encodedDimension
        self.retrainingHealthProfilePath = np.zeros(shape=(numProfileShots + 1, 1, self.encodedDimension), dtype=np.float16)  # Dim: numProfileShots, numExperiments, encodedDimension
        self.retrainingProfileLosses = np.zeros(shape=(numProfileShots + 1, self.numExperiments, self.numSignals), dtype=np.float16)

    def resetModelStates(self, metaLearningData):
        with torch.no_grad():
            # Pre-allocate each parameter.
            numExperiments, numSignals, encodedDimension = metaLearningData.shape
            numSpecificEncoderLayers, numSharedEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers'], modelConstants.userInputParams['numSharedEncoderLayers']
            self.compiledLayerStates = np.zeros(shape=(numSpecificEncoderLayers + numSharedEncoderLayers + 1, numExperiments, numSignals, encodedDimension), dtype=np.float16)
            self.compiledLayerStateInd = 0

            # Add the initial state.
            self.addModelState(metaLearningData)

    def addModelState(self, metaLearningData):
        self.compiledLayerStates[self.compiledLayerStateInd] = metaLearningData.view((self.compiledLayerStates.shape[1], self.compiledLayerStates.shape[2], self.compiledLayerStates.shape[3])).detach().clone().cpu().numpy().astype(np.float16)
        self.compiledLayerStateInd += 1

    def populateProfileState(self, profileEpoch, batchInds, profileStateLoss, resampledSignalData, healthProfile):
        with torch.no_grad():
            if isinstance(batchInds, torch.Tensor): batchInds = batchInds.detach().cpu().numpy().astype(np.int32)
            self.retrainingProfileLosses[profileEpoch][batchInds] = profileStateLoss.detach().cpu().numpy().astype(np.float16)

            if 0 not in batchInds: return None
            # For space efficiency, only store the first batch and signal.
            self.generatingBiometricSignals[profileEpoch][0] = resampledSignalData[batchInds == 0, :].detach().cpu().numpy().astype(np.float16)
            self.retrainingHealthProfilePath[profileEpoch][0] = healthProfile[batchInds == 0].detach().cpu().numpy().astype(np.float16)

    def getHealthEmbedding(self, batchInds, fourierModel):
        imaginaryProfileCoefficients = self.imaginaryProfileCoefficients.to(batchInds.device)[batchInds].unsqueeze(1)
        realProfileCoefficients = self.realProfileCoefficients.to(batchInds.device)[batchInds].unsqueeze(1)
        # ...ProfileCoefficients dimension: numExperiments, 1, fourierDimension

        # Learn the health profile.
        coefficients = torch.cat(tensors=[realProfileCoefficients, imaginaryProfileCoefficients], dim=1)
        coefficients = fourierModel(coefficients)

        # Return to health profile.
        return torch.fft.irfft(coefficients[:, 0, :] + 1j * coefficients[:, 1, :], n=self.encodedDimension, dim=-1, norm='ortho')
