# Import helper files.
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class modelParameters:

    def __init__(self, userInputParams, accelerator=None):
        # General parameters
        self.userInputParams = userInputParams  # The user input parameters.
        self.accelerator = accelerator  # The single-instance accelerator for the model.

        # Run-specific parameters.
        self.hpcTrialRun = userInputParams['deviceListed'].startswith("HPC") if userInputParams else False  # The HPC trial run flag.
        self.gpuFlag = accelerator.device.type == 'cuda' if accelerator else False  # The GPU flag.

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
                'skipConnectionProtocol': 'none',  # The protocol for the skip connections.
                'learningProtocol': 'rFC'  # The learning protocol for the neural operator.
            }
        }

        return userInputParams

    def getTrainingBatchSize(self, submodel, numExperiments):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 60 experiments with 69 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 707 experiments with 126 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 22 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1460 experiments with 51 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 184 experiments with 80 signals.
        if submodel == modelConstants.signalEncoderModel: totalMinBatchSize = 32
        elif submodel == modelConstants.emotionModel: totalMinBatchSize = 32
        else: raise Exception()

        # Adjust the batch size based on the number of gradient accumulations.
        gradientAccumulation = self.accelerator.gradient_accumulation_steps
        minimumBatchSize = totalMinBatchSize // gradientAccumulation
        # Assert that the batch size is divisible by the gradient accumulation steps.
        assert totalMinBatchSize % gradientAccumulation == 0, "The total batch size must be divisible by the gradient accumulation steps."
        assert gradientAccumulation <= totalMinBatchSize, "The gradient accumulation steps must be less than the total batch size."

        # Adjust the batch size based on the total size.
        batchSize = int(minimumBatchSize * numExperiments / modelConstants.minNumExperiments)
        batchSize = min(batchSize, numExperiments)

        return batchSize

    @staticmethod
    def getInferenceBatchSize(submodel, device):
        if submodel == modelConstants.signalEncoderModel: return 32 if device == "cpu" else 16
        elif submodel == modelConstants.emotionModel: return 32 if device == "cpu" else 16
        else: raise Exception()

    @staticmethod
    def getEpochInfo(useFinalParams):
        return (1000, 10, 10) if useFinalParams else (-1, -1, -1)  # numEpochs, numEpoch_toPlot, numEpoch_toSaveFull

    # -------------------------- Compilation Parameters ------------------------- #

    @staticmethod
    def getExclusionCriteria(submodel):
        if submodel == modelConstants.signalEncoderModel:
            return -1, 2  # Emotion classes dont matter.
        elif submodel == modelConstants.emotionModel:
            return 2, 0.8
        else:
            raise Exception()

    # -------------------------- Saving/Loading Parameters ------------------------- #

    @staticmethod
    def getModelInfo(submodel, specificInfo=None):
        # Base case: information hard-coded.
        if specificInfo is not None:
            return specificInfo

        # No model information to load.
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = None, None, None

        if submodel == modelConstants.emotionModel:
            # Model loading information.
            loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
            loadSubmodel = modelConstants.signalEncoderModel  # The submodel to load.
            loadSubmodelEpochs = -1  # The # of epochs to load from the trained model.

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
            submodelsSaving = [modelConstants.trainingInformation, modelConstants.specificSignalEncoderModel, modelConstants.sharedSignalEncoderModel]
        elif submodel == modelConstants.emotionModel:
            submodelsSaving = [modelConstants.trainingInformation, modelConstants.specificSignalEncoderModel, modelConstants.sharedSignalEncoderModel, modelConstants.specificEmotionModel, modelConstants.sharedEmotionModel]
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
