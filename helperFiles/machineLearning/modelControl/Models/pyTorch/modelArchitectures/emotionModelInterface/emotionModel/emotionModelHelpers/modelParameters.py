import random

# Import helper files.
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods


class modelParameters:

    def __init__(self, userInputParams, accelerator=None):
        # General parameters
        self.userInputParams = userInputParams  # The user input parameters.
        self.accelerator = accelerator  # The single-instance accelerator for the model.

        # Run-specific parameters.
        self.hpcTrialRun = userInputParams['deviceListed'].startswith("HPC") if userInputParams else False  # The HPC trial run flag.
        self.gpuFlag = accelerator.device.type == 'cuda' if accelerator else False  # The GPU flag.

        # General parameters
        self.demographicIdentifiers = []  # The demographic identifiers to consider.
        self.subjectI = []  # The emotion identifiers to consider.`
        self.timeWindows = self.getTimeWindows()  # The time windows to consider.
        self.maxNumSignals = 138  # The maximum number of signals to consider.
        self.minTimeBuffer = 600  # The minimum time buffer to consider.
        self.minFeatureFreq = 0.1  # The minimum average number of sequence points/second to consider.

        # Helper classes.
        self.generalMethods = generalMethods()

    # -------------------------- Training Parameters ------------------------- #

    def getAugmentationDeviation(self, submodel):
        # Get the submodels to save
        if submodel == "signalEncoder":
            addingNoiseRange = (0, 0.01)
        elif submodel == "autoencoder":
            addingNoiseRange = (0, 0.01)
        elif submodel == "emotionPrediction":
            addingNoiseRange = (0, 0.01)
        else:
            assert False, "No model initialized"

        return self.generalMethods.biased_high_sample(*addingNoiseRange, randomValue=random.uniform(a=0, b=1)), addingNoiseRange

    def getTrainingBatchSize(self, submodel, numExperiments):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 61 experiments with 70 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 318 experiments with 120 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 19 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1523 experiments with 49 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 177 experiments with 79 signals.

        if submodel == "signalEncoder":
            totalMinBatchSize = 16
        elif submodel == "autoencoder":
            totalMinBatchSize = 16
        elif submodel == "emotionPrediction":
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
        batchSize = int(minimumBatchSize * numExperiments / 61)
        batchSize = min(batchSize, numExperiments)

        return batchSize

    def getInferenceBatchSize(self, submodel, numSignals):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 61 experiments with 70 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 55 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 318 experiments with 120 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 19 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1523 experiments with 49 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 177 experiments with 79 signals.
        # Set the minimum batch size.
        minimumBatchSize = 32 if self.gpuFlag else 16

        if submodel == "signalEncoder":
            if self.userInputParams['numSigEncodingLayers'] <= 2 and self.userInputParams['numSigLiftedChannels'] <= 16: minimumBatchSize = 64
        elif submodel == "autoencoder":
            minimumBatchSize = 32 if self.hpcTrialRun else 32
        elif submodel == "emotionPrediction":
            minimumBatchSize = 32 if self.hpcTrialRun else 32
        else:
            raise Exception()

        # Adjust the batch size based on the number of signals used.
        maxBatchSize = int(minimumBatchSize * self.maxNumSignals / numSignals)
        maxBatchSize = min(maxBatchSize, numSignals)  # Ensure the maximum batch size is not larger than the number of signals.

        return maxBatchSize

    @staticmethod
    def getNumEpochs(submodel):
        if submodel == "signalEncoder":
            return 10, 1000  # numConstrainedEpochs, numEpoch
        elif submodel == "autoencoder":
            return 10, 1000  # numConstrainedEpochs, numEpoch
        elif submodel == "emotionPrediction":
            return 10, 1000  # numConstrainedEpochs, numEpoch
        else:
            raise Exception()

    @staticmethod
    def getEpochInfo(submodel, useFinalParams):
        if submodel == "signalEncoder":
            return 10, 10 if useFinalParams else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == "autoencoder":
            return 10, 10 if useFinalParams else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == "emotionPrediction":
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

    def getFinalDistributionLength(self):
        return self.timeWindows[-1]

    def getMaxBufferLength(self):
        maxTimeBufferConsidered = 2 * self.getTimeWindows()[-1] + self.getShiftInfo(submodel='maxShift')
        modelFeatureTimeBuffer = max(self.minTimeBuffer, maxTimeBufferConsidered)

        return modelFeatureTimeBuffer

    @staticmethod
    def getSignalMinMaxScale():
        return 1  # Some wavelets constrained to +/- 1.

    @staticmethod
    def getTimeWindows():
        return [90, 120, 150, 180, 210, 240, 300]

    def getSequenceLengthRange(self, submodel, sequenceLength):
        if submodel == "signalEncoder":
            return self.timeWindows[0], self.timeWindows[-1]
        elif submodel == "autoencoder":
            return self.timeWindows[0], self.timeWindows[-1]
        elif submodel == "emotionPrediction":
            assert self.timeWindows[0] <= sequenceLength <= self.timeWindows[-1], "The sequence length must be within the trained time windows."
            return sequenceLength, sequenceLength
        else:
            raise Exception()

    @staticmethod
    def getShiftInfo(submodel):
        if submodel == "signalEncoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 60, 30  # dontShiftDatasets, numSecondsShift, numSeconds_perShift
        elif submodel == "autoencoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 60, 30  # dontShiftDatasets, numSecondsShift, numSeconds_perShift
        elif submodel == "emotionPrediction":
            return ['case', 'amigos'], 5, 2  # dontShiftDatasets, numSecondsShift, numSeconds_perShift
        elif submodel == "maxShift":
            return 60
        else:
            raise Exception()

    @staticmethod
    def getExclusionCriteria(submodel):
        if submodel == "signalEncoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "autoencoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "emotionPrediction":
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

        if submodel == "autoencoder":
            # Model loading information.
            loadSubmodelDate = f"2024-04-06 Final signalEncoder on cuda at numExpandedSignals 4 at numSigEncodingLayers 4"  # The date the model was trained.
            loadSubmodel = "signalEncoder"  # The model's component we are loading.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.

        elif submodel == "emotionPrediction":
            # Model loading information.
            loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
            loadSubmodel = "autoencoder"  # The model's component we are loading.
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
        if submodel == "signalEncoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel"]
        elif submodel == "autoencoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel"]
        elif submodel == "emotionPrediction":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
        else:
            assert False, "No model initialized"

        return submodelsSaving

    @staticmethod
    def getSharedModels():
        # Possible models: ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
        sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]

        return sharedModelWeights

    # -------------------------- Organizational Methods ------------------------- #

    @staticmethod
    def compileModelNames():
        # Specify which metadata analyses to compile
        metaDatasetNames = ["wesad", "emognition", "amigos", "dapper", "case"]
        datasetNames = ['empatch']
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
