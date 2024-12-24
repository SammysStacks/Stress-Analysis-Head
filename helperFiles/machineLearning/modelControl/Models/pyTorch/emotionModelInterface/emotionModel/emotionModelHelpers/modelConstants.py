import torch


class modelConstants:
    # -------------------------- Dataset Constants ------------------------- #

    # Specify the dataset names
    emognitionDatasetName = "emognition"
    empaticaDatasetName = "empatica"
    empatchDatasetName = "empatch"
    amigosDatasetName = "amigos"
    dapperDatasetName = "dapper"
    wesadDatasetName = "wesad"
    caseDatasetName = "case"

    # -------------------------- Model Constants ------------------------- #

    # Specify the submodels.
    specificSignalEncoderModel = "specificSignalEncoderModel"
    sharedSignalEncoderModel = "sharedSignalEncoderModel"
    specificActivityModel = "specificActivityModel"
    specificEmotionModel = "specificEmotionModel"
    sharedActivityModel = "sharedActivityModel"
    sharedEmotionModel = "sharedEmotionModel"

    # Specify the models.
    signalEncoderModel = "signalEncoderModel"
    modelName = "observationModel"
    emotionModel = "emotionModel"

    # Specify the subject identifiers.
    numSignalPointsSI = "numSignalPoints"
    subjectIndexMD = "subjectIndex"
    datasetIndexMD = "datasetIndex"
    signalIndexSI = "signalIndex"
    batchIndexSI = "batchIndex"

    # Specify the channel information.
    signalChannel = "signalPoints"
    timeChannel = "time"

    # Compile the contextual identifiers.
    signalIdentifiers = [signalIndexSI, batchIndexSI]
    metadata = [datasetIndexMD, subjectIndexMD]

    # Specify the model parameters.
    initialProfileAmp = None  # The limits for the uniform initialization.
    numEncodedWeights = None  # The number of encoded weights.
    modelTimeWindow = 120  # The time window for the model.
    numEpochs_minLR = 0  # The number of warmup epochs. 4
    minMaxScale = 1  # The maximum value for the min-max scaling.
    numBatches = 16  # The number of batches to use in the model.
    numWarmups = 0  # The number of warmup epochs. 5

    # Specify the data interface parameter names.
    signalChannelNames = [timeChannel, signalChannel]

    # Specify the saving parameters.
    specificModelWeights = [specificSignalEncoderModel, specificEmotionModel, specificActivityModel]
    sharedModelWeights = [sharedSignalEncoderModel, sharedEmotionModel, sharedActivityModel]
    userInputParams = {}

    @classmethod
    def updateModelParams(cls, userInputParams):
        cls.userInputParams = userInputParams

        # Update the model constants.
        modelConstants.initialProfileAmp = userInputParams['initialProfileAmp']
        modelConstants.numEncodedWeights = userInputParams['numEncodedWeights']

    # ---------------- Hard-coded therapy parameters --------------------- #

    therapyParams = {
        'heat': [torch.full(size=(1, 1, 1, 1), fill_value=37)],
        'music': [440, 410],
        'Vr': None
    }

    # -------------------------------------------------------------------- #
