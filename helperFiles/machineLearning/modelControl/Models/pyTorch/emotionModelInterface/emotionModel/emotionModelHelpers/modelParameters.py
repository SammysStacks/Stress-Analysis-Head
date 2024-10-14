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
                'learningProtocol': 'iCNN'  # The learning protocol for the neural operator.
            }
        }

        return userInputParams

    def getTrainingBatchSize(self, submodel, numExperiments):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 59 experiments with 68 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 707 experiments with 127 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 22 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1584 experiments with 51 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 154 experiments with 81 signals.

        if submodel == modelConstants.signalEncoderModel:
            totalMinBatchSize = 16
        elif submodel == modelConstants.emotionModel:
            totalMinBatchSize = 16
        else:
            raise Exception()

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

    def getInferenceBatchSize(self, submodel, numSignals):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 59 experiments with 68 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 707 experiments with 127 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 22 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1584 experiments with 51 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 154 experiments with 81 signals.
        # Set the minimum batch size.
        if submodel == modelConstants.signalEncoderModel:
            minimumBatchSize = 32 if self.gpuFlag else 32
        elif submodel == modelConstants.emotionModel:
            minimumBatchSize = 32 if self.gpuFlag else 32
        else:
            raise Exception()

        # Adjust the batch size based on the number of signals used.
        maxBatchSize = int(minimumBatchSize * modelConstants.maxNumSignals / numSignals)
        maxBatchSize = min(maxBatchSize, numSignals)  # Ensure the maximum batch size is not larger than the number of signals.

        return maxBatchSize

    @staticmethod
    def getEpochInfo(useFinalParams):
        return (10, 10, 10) if useFinalParams else (-1, -1, -1)  # numEpochs, numEpoch_toPlot, numEpoch_toSaveFull

    # -------------------------- Compilation Parameters ------------------------- #

    @staticmethod
    def getSequenceLengthRange(submodel, sequenceLength):
        if submodel == modelConstants.signalEncoderModel:
            return modelConstants.timeWindows[0], modelConstants.timeWindows[-1]
        elif submodel == modelConstants.emotionModel:
            assert modelConstants.timeWindows[0] <= sequenceLength <= modelConstants.timeWindows[-1], "The sequence length must be within the trained time windows."
            return sequenceLength, sequenceLength
        else:
            raise Exception()

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
