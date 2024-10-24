
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
    inferenceModel = "inferenceModel"

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
    signalIdentifiers = [numSignalPointsSI, signalIndexSI, batchIndexSI]
    metadata = [datasetIndexMD, subjectIndexMD]

    # Specify the data interface parameters.
    timeWindows = [90, 120, 150, 180, 210, 240]
    minMaxScale = 1  # The maximum value for the min-max scaling.
    numBatches = 15  # The number of batches to use in the model.

    # Specify the data interface parameter names.
    signalChannelNames = [timeChannel, signalChannel]

    # Specify the saving parameters.
    specificModelWeights = [specificSignalEncoderModel, specificEmotionModel]
    sharedModelWeights = [sharedSignalEncoderModel, sharedEmotionModel]
    inferenceModelWeights = [inferenceModel]

    # -------------------------------------------------------------------- #
