# Import helper files.
import math

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class modelParameters:

    def __init__(self, accelerator=None):
        # General parameters
        self.accelerator = accelerator  # The single-instance accelerator for the model.

        # Helper classes.
        self.generalMethods = generalMethods()

    # -------------------------- Training Parameters ------------------------- #

    @staticmethod
    def getNeuralParameters(userInputParams):
        userInputParams['neuralOperatorParameters'] = {
            'wavelet': {
                'waveletType': userInputParams['waveletType'],  # The type of wavelet to use for the wavelet transform.
                'encodeHighFrequencyProtocol': 'highFreq',  # The protocol for encoding the high frequency signals.
                'encodeLowFrequencyProtocol': 'lowFreq',  # The protocol for encoding the low frequency signals.
            },
            'fourier': {
                'encodeImaginaryFrequencies': True,  # The protocol for encoding the high frequency signals.
                'encodeRealFrequencies': True,  # The protocol for encoding the low frequency signals.
            }
        }

        # Delete the waveletType from the user input parameters.
        del userInputParams['waveletType']

        return userInputParams

    def getTrainingBatchSize(self, submodel, numExperiments):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 60 experiments with 29 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 47 signals.
        # Amigos: Found 10 (out of 12) well-labeled emotions across 673 experiments with 118 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 15 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1442 experiments with 42 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 165 experiments with 48 signals.
        if submodel == modelConstants.signalEncoderModel: effectiveMinBatchSize, effectiveMaxBatchSize = 12, 160
        elif submodel == modelConstants.emotionModel: effectiveMinBatchSize, effectiveMaxBatchSize = 12, 160
        else: raise Exception()

        # Adjust the batch size based on the number of gradient accumulations.
        gradientAccumulation = self.accelerator.gradient_accumulation_steps
        minBatchSize_perLoop = effectiveMinBatchSize / gradientAccumulation
        maxBatchSize_perLoop = effectiveMaxBatchSize / gradientAccumulation
        metaBatchSize = numExperiments / modelConstants.numBatches / gradientAccumulation
        # Assert that the batch size is divisible by the gradient accumulation steps.
        assert effectiveMinBatchSize % gradientAccumulation == 0, "The total batch size must be divisible by the gradient accumulation steps."
        assert gradientAccumulation <= effectiveMinBatchSize, "The gradient accumulation steps must be less than the total batch size."

        # Adjust the batch size based on the data ratio.
        batchSize = max(metaBatchSize, minBatchSize_perLoop)
        batchSize = math.ceil(min(batchSize, maxBatchSize_perLoop))
        batchSize = batchSize + batchSize % gradientAccumulation

        return batchSize

    @staticmethod
    def getInferenceBatchSize(submodel, device):
        if submodel == modelConstants.signalEncoderModel: return 256 if device == "cpu" else 256
        elif submodel == modelConstants.emotionModel: return 256 if device == "cpu" else 256
        else: raise Exception()

    @staticmethod
    def getEpochInfo():
        return 2500, 10, 10  # numEpochs, numEpoch_toPlot, numEpoch_toSaveFull

    # -------------------------- Compilation Parameters ------------------------- #

    @staticmethod
    def getExclusionSequenceCriteria():
        return 30, 30, 2, 0.25, 0.1  # minSequencePoints, minSignalPresentCount, minBoundaryPoints, maxSinglePointDiff, maxAverageDiff

    @staticmethod
    def getAdaptiveFactor():
        return 2

    # -------------------------- Saving/Loading Parameters ------------------------- #

    @staticmethod
    def getModelInfo(submodel, specificInfo=None):
        # Base case: information hard-coded.
        if specificInfo is not None:
            return specificInfo

        # No model information to load.
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = None, None, None

        # if submodel == modelConstants.emotionModel:
        #     # Model loading information.
        #     loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
        #     loadSubmodel = modelConstants.signalEncoderModel  # The submodel to load.
        #     loadSubmodelEpochs = -1  # The # of epochs to load from the trained model.

        return loadSubmodelDate, loadSubmodelEpochs, loadSubmodel

    @staticmethod
    def getSavingInformation(epoch, numEpoch_toSaveFull, numEpoch_toPlot):
        saveFullModel = (epoch % numEpoch_toSaveFull == 0)
        plotSteps = (epoch % numEpoch_toPlot == 0)

        return saveFullModel, plotSteps

    @staticmethod
    def getSubmodelsSaving(submodel):
        # Get the submodels to save
        if submodel == modelConstants.signalEncoderModel:
            submodelsSaving = [modelConstants.specificSignalEncoderModel, modelConstants.sharedSignalEncoderModel]
        elif submodel == modelConstants.emotionModel:
            submodelsSaving = [modelConstants.specificSignalEncoderModel, modelConstants.sharedSignalEncoderModel, modelConstants.specificEmotionModel, modelConstants.sharedEmotionModel, modelConstants.specificActivityModel, modelConstants.sharedActivityModel]
        else: assert False, "No model initialized"

        return submodelsSaving

    # -------------------------- Organizational Methods ------------------------- #

    @staticmethod
    def compileModelNames():
        # Specify which metadata analyses to compile
        metaDatasetNames = [modelConstants.wesadDatasetName, modelConstants.emognitionDatasetName, modelConstants.amigosDatasetName, modelConstants.dapperDatasetName, modelConstants.caseDatasetName]
        datasetNames = [modelConstants.empatchDatasetName]
        allDatasetNames = metaDatasetNames + datasetNames

        # Assert the integrity of dataset collection.
        assert len(datasetNames) == 1

        return datasetNames, metaDatasetNames, allDatasetNames
