import random

# Import helper files.
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


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

    def getAugmentationDeviation(self, submodel):
        if submodel == modelConstants.emotionPredictionModel:
            addingNoiseRange = (0, 0.01)
            return self.generalMethods.biased_high_sample(*addingNoiseRange, randomValue=random.uniform(a=0, b=1)), addingNoiseRange

        return 0, (0, 1)

    def getTrainingBatchSize(self, submodel, numExperiments):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 59 experiments with 68 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 707 experiments with 127 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 22 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1584 experiments with 51 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 154 experiments with 81 signals.

        if submodel == modelConstants.signalEncoderModel:
            totalMinBatchSize = 16
        elif submodel == modelConstants.autoencoderModel:
            totalMinBatchSize = 16
        elif submodel == modelConstants.emotionPredictionModel:
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
        minimumBatchSize = 32 if self.gpuFlag else 16

        if submodel == modelConstants.signalEncoderModel:
            if self.userInputParams['numSigEncodingLayers'] <= 2 and self.userInputParams['numSigLiftedChannels'] <= 16: minimumBatchSize = 64
        elif submodel == modelConstants.autoencoderModel:
            minimumBatchSize = 32 if self.hpcTrialRun else 32
        elif submodel == modelConstants.emotionPredictionModel:
            minimumBatchSize = 32 if self.hpcTrialRun else 32
        else:
            raise Exception()

        # Adjust the batch size based on the number of signals used.
        maxBatchSize = int(minimumBatchSize * modelConstants.maxNumSignals / numSignals)
        maxBatchSize = min(maxBatchSize, numSignals)  # Ensure the maximum batch size is not larger than the number of signals.

        return maxBatchSize

    @staticmethod
    def getNumEpochs(submodel):
        if submodel == modelConstants.signalEncoderModel:
            return 10, 1000  # numConstrainedEpochs, numEpoch
        elif submodel == modelConstants.autoencoderModel:
            return 10, 1000  # numConstrainedEpochs, numEpoch
        elif submodel == modelConstants.emotionPredictionModel:
            return 10, 1000  # numConstrainedEpochs, numEpoch
        else:
            raise Exception()

    @staticmethod
    def getEpochInfo(submodel, useFinalParams):
        if submodel == modelConstants.signalEncoderModel:
            return 10, 10 if useFinalParams else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == modelConstants.autoencoderModel:
            return 10, 10 if useFinalParams else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == modelConstants.emotionPredictionModel:
            return 10, 10 if useFinalParams else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        else:
            raise Exception()

    def alterProtocolParams(self, storeLoss, fastPass, useFinalParams):
        # Self-check the hpc parameters.
        if self.hpcTrialRun and useFinalParams:
            self.accelerator.gradient_accumulation_steps = 16
            storeLoss = True  # Turn on loss storage for HPC.
            fastPass = False  # Turn off fast pass for HPC.

        # Set CPU settings.
        if not self.gpuFlag:
            self.accelerator.gradient_accumulation_steps = 16

        # Relay the inputs to the user.
        numGradientSteps = self.accelerator.gradient_accumulation_steps
        print(f"Final parameters: storeLoss={storeLoss}, fastPass={fastPass}, device={self.accelerator.device}, numGradientSteps={numGradientSteps}", flush=True)

        return storeLoss, fastPass

    # -------------------------- Compilation Parameters ------------------------- #

    @staticmethod
    def getSequenceLengthRange(submodel, sequenceLength):
        if submodel == modelConstants.signalEncoderModel:
            return modelConstants.timeWindows[0], modelConstants.timeWindows[-1]
        elif submodel == modelConstants.autoencoderModel:
            return modelConstants.timeWindows[0], modelConstants.timeWindows[-1]
        elif submodel == modelConstants.emotionPredictionModel:
            assert modelConstants.timeWindows[0] <= sequenceLength <= modelConstants.timeWindows[-1], "The sequence length must be within the trained time windows."
            return sequenceLength, sequenceLength
        else:
            raise Exception()

    @staticmethod
    def getExclusionCriteria(submodel):
        if submodel == modelConstants.signalEncoderModel:
            return -1, 2  # Emotion classes dont matter.
        elif submodel == modelConstants.autoencoderModel:
            return -1, 2  # Emotion classes dont matter.
        elif submodel == modelConstants.emotionPredictionModel:
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
        loadSubmodelEpochs = None
        loadSubmodelDate = None
        loadSubmodel = None

        if submodel == modelConstants.autoencoderModel:
            # Model loading information.
            loadSubmodelDate = f"2024-04-06 Final signalEncoder on cuda at numExpandedSignals 4 at numSigEncodingLayers 4"  # The date the model was trained.
            loadSubmodel = modelConstants.signalEncoderModel  # The model's component we are loading.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.

        elif submodel == modelConstants.emotionPredictionModel:
            # Model loading information.
            loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
            loadSubmodel = modelConstants.autoencoderModel  # The model's component we are loading.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.

        return loadSubmodelDate, loadSubmodelEpochs, loadSubmodel

    @staticmethod
    def getSavingInformation(epoch, numConstrainedEpochs, numEpoch_toSaveFull, numEpoch_toPlot):
        # Initialize flags to False.
        saveFullModel = False

        # Determine if we should save or plot the model.
        if epoch <= numConstrainedEpochs:
            plotSteps = True
        else:
            saveFullModel = (epoch % numEpoch_toSaveFull == 0)
            plotSteps = (epoch % numEpoch_toPlot == 0)

        return saveFullModel, plotSteps

    @staticmethod
    def getSubmodelsSaving(submodel):
        # Get the submodels to save
        if submodel == modelConstants.signalEncoderModel:
            submodelsSaving = [modelConstants.trainingInformation, modelConstants.signalEncoderModel]
        elif submodel == modelConstants.autoencoderModel:
            submodelsSaving = [modelConstants.trainingInformation, modelConstants.signalEncoderModel, modelConstants.autoencoderModel]
        elif submodel == modelConstants.emotionPredictionModel:
            submodelsSaving = [modelConstants.trainingInformation, modelConstants.signalEncoderModel, modelConstants.autoencoderModel, modelConstants.signalMappingModel, modelConstants.specificEmotionModel, modelConstants.sharedEmotionModel]
        else:
            assert False, "No model initialized"

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

    @staticmethod
    def compileParameters(args):
        # Organize the input information into a dictionary.
        userInputParams = {
            # Assign general model parameters
            'optimizerType': args.optimizerType,  # The optimizerType used during training convergence.
            'deviceListed': args.deviceListed,  # The device we are running the platform on.
            'submodel': args.submodel,  # The component of the model we are training.
            # Assign signal encoder parameters
            'signalEncoderWaveletType': args.signalEncoderWaveletType,  # The wavelet type for the wavelet transform.
            'numSigLiftedChannels': args.numSigLiftedChannels,  # The number of channels to lift to during signa; encoding.
            'numSigEncodingLayers': args.numSigEncodingLayers,  # The number of operator layers during signal encoding.
            'numExpandedSignals': args.numExpandedSignals,  # The number of signals to group when you begin compression or finish expansion.
            # Assign autoencoder parameters
            'compressionFactor': args.compressionFactor,  # The compression factor of the autoencoder.
            'expansionFactor': args.expansionFactor,  # The expansion factor of the autoencoder.
            # Assign emotion prediction parameters
            'numInterpreterHeads': args.numInterpreterHeads,  # The number of ways to interpret a set of physiological signals.
            'numBasicEmotions': args.numBasicEmotions,  # The number of basic emotions (basis states of emotions).
            'finalDistributionLength': args.finalDistributionLength,  # The maximum number of time series points to consider.
        }

        # Relay the inputs to the user.
        print("System Arguments:", userInputParams, flush=True)
        submodel = args.submodel

        # Assert the integrity of the model parameters.
        assert args.numExpandedSignals <= args.numSigLiftedChannels

        return userInputParams, submodel
