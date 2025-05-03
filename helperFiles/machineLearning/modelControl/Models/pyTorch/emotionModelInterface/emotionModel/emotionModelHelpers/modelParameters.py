# Import helper files.
import math

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class modelParameters:

    def __init__(self, accelerator=None):
        # General parameters
        self.accelerator = accelerator  # The single-instance accelerator for the model.
        self.generalMethods = generalMethods()

    # -------------------------- Training Parameters ------------------------- #

    @staticmethod
    def getNeuralParameters(userInputParams):
        userInputParams['neuralOperatorParameters'] = {
            'wavelet': {
                'minWaveletDim': userInputParams['minWaveletDim'],  # The minimum dimension for the wavelet transform.
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

    def getTrainingBatchSize(self, submodel, numExperiments, datasetName):
        """
            Wesad: Found 32 (out of 32) emotions across 60 experiments for 28 signals with 2.0 batches of 30 experiments
            Amigos: Found 12 (out of 12) emotions across 673 experiments for 60 signals with 11.807 batches of 57 experiments
            Dapper: Found 12 (out of 12) emotions across 364 experiments for 15 signals with 11.742 batches of 31 experiments
            Case: Found 2 (out of 2) emotions across 1442 experiments for 35 signals with 11.917 batches of 121 experiments
            Emognition: Found 12 (out of 12) emotions across 407 experiments for 39 signals with 11.971 batches of 34 experiments
            Empatch: Found 30 (out of 30) emotions across 165 experiments for 28 signals with 5.0 batches of 33 experiments
        """
        if submodel == modelConstants.signalEncoderModel: effectiveMinBatchSize, effectiveMaxBatchSize = 24, 128
        elif submodel == modelConstants.emotionModel: effectiveMinBatchSize, effectiveMaxBatchSize = 24, 128
        else: raise Exception()

        if datasetName == modelConstants.wesadDatasetName: effectiveMinBatchSize = 20
        if datasetName == modelConstants.empatchDatasetName: effectiveMinBatchSize = 55
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
    def getEpochInfo(validationRun):
        if validationRun: return 2001, 50, 50  # loadSubmodelEpochs, numEpoch_toPlot, numEpoch_toSaveFull
        else: return 2001, 50, 50  # loadSubmodelEpochs, numEpoch_toPlot, numEpoch_toSaveFull

    @staticmethod
    def getProfileEpochs(): return modelConstants.userInputParams['numProfileShots']  # loadSubmodelEpochs

    @staticmethod
    def getEpochParameters(epoch, numEpoch_toSaveFull, numEpoch_toPlot, plotAllEpochs):
        saveFullModel = (epoch % numEpoch_toSaveFull == 0)
        plotSteps = (epoch % numEpoch_toPlot == 0)
        if plotAllEpochs: plotSteps = True

        return saveFullModel, plotSteps

    # -------------------------- Compilation Parameters ------------------------- #

    @staticmethod
    def getExclusionSequenceCriteria(): return 32, 32, 2, 0.25, 0.15  # minSequencePoints, minSignalPresentCount, minBoundaryPoints, maxSinglePointDiff, maxAverageDiff

    @staticmethod
    def embedInformation(submodel, trainingDate, validationRun):
        # Embedded information for each model.
        userInputParams = modelConstants.userInputParams
        if userInputParams['encodedDimension'] < userInputParams['profileDimension']: raise Exception("The number of encoded weights must be less than the encoded dimension.")

        # Get the model information.
        signalEncoderModelInfo = (f"signalEncoder{' validation' if validationRun else ''} shared-{userInputParams['numIgnoredSharedHF']} thresh{round(float(userInputParams['minAngularThreshold']), 1)}-{userInputParams['minThresholdStep']}-{round(float(userInputParams['maxAngularThreshold']), 1)} {userInputParams['optimizerType']} {userInputParams['numSharedEncoderLayers']}-shared specific-{userInputParams['numSpecificEncoderLayers']} " +
                                  f"LR{userInputParams['profileLR']}-{userInputParams['physGenLR']}-{round(userInputParams['reversibleLR']*180/math.pi, 3)} profileParams{userInputParams['profileDimension']} numShots{userInputParams['numProfileShots']} encodedDim{userInputParams['encodedDimension']} {userInputParams['neuralOperatorParameters']['wavelet']['waveletType']}-{userInputParams['minWaveletDim']}")
        emotionModelInfo = (f"emotionPrediction{' validation' if validationRun else ''} shared-{userInputParams['numIgnoredSharedHF']} thresh{round(float(userInputParams['minAngularThreshold']), 1)}-{userInputParams['minThresholdStep']}-{round(float(userInputParams['maxAngularThreshold']), 1)} {userInputParams['optimizerType']} {userInputParams['numSharedEncoderLayers']}-shared specific-{userInputParams['numSpecificEncoderLayers']} " +
                            f"LR{userInputParams['profileLR']}-{userInputParams['physGenLR']}-{round(userInputParams['reversibleLR']*180/math.pi, 3)} profileParams{userInputParams['profileDimension']} numShots{userInputParams['numProfileShots']} encodedDim{userInputParams['encodedDimension']} {userInputParams['neuralOperatorParameters']['wavelet']['waveletType']}-{userInputParams['minWaveletDim']}")

        # Return the model information.
        if submodel == modelConstants.signalEncoderModel: return f"{trainingDate} {signalEncoderModelInfo.replace('.', '-')}"
        elif submodel == modelConstants.emotionModel: return f"{trainingDate} {emotionModelInfo.replace('.', '-')}"
        else: raise Exception()

    # -------------------------- Saving/Loading Parameters ------------------------- #

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
        metaDatasetNames = [modelConstants.wesadDatasetName, modelConstants.amigosDatasetName, modelConstants.dapperDatasetName, modelConstants.caseDatasetName, modelConstants.emognitionDatasetName]
        datasetNames = [modelConstants.empatchDatasetName]

        # Assert the integrity of dataset collection.
        assert len(datasetNames) == 1

        return datasetNames, metaDatasetNames
