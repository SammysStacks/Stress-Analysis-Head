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

    timeWindows = [90, 120]
    # Specify the data interface parameters.
    numEncodedWeights = None  # The number of encoded weights.
    minMaxScale = 1  # The maximum value for the min-max scaling.
    numBatches = 16  # The number of batches to use in the model.

    # Specify the data interface parameter names.
    signalChannelNames = [timeChannel, signalChannel]

    # Specify the saving parameters.
    specificModelWeights = [specificSignalEncoderModel, specificEmotionModel, specificActivityModel]
    sharedModelWeights = [sharedSignalEncoderModel, sharedEmotionModel, sharedActivityModel]
    userInputParams = {'numProfileEpochs': 30}

    @classmethod
    def updateModelParams(cls, userInputParams):
        cls.userInputParams = userInputParams

        # Update the model constants.
        modelConstants.numEncodedWeights = userInputParams['numEncodedWeights']

    # ---------------- Hard-coded therapy parameters --------------------- #

    therapyParams = {
        'heat': [torch.full(size=(1, 1, 1, 1), fill_value=37)],
        'music': [440, 410],
        'Vr': None
    }

    # -------------------------------------------------------------------- #

