
class modelConstants:

    # -------------------------- Dataset Constants ------------------------- #

    # Specify the dataset names
    emognitionDatasetName = "emognition"
    empatchDatasetName = "empatch"
    amigosDatasetName = "amigos"
    dapperDatasetName = "dapper"
    wesadDatasetName = "wesad"
    caseDatasetName = "case"

    # -------------------------- Model Constants ------------------------- #

    # Specify the submodels.
    emotionPredictionModel = "emotionPredictionModel"
    signalEncoderModel = "signalEncoderModel"
    autoencoderModel = "autoencoderModel"

    # Specify the subject identifiers.
    stopSignalIndexSI = "stopSignalIndex"
    subjectIndexSI = "subjectIndex"
    datasetIndexSI = "datasetIndex"

    # Compile the subject identifiers.
    subjectIdentifiers = [stopSignalIndexSI, subjectIndexSI, datasetIndexSI]

    # Specify the data interface parameters.
    timeWindows = [90, 120, 150, 180, 210, 240, 300]
    finalDistributionLength = 300
    numSignalChannels = 3  # Total channels is numSignalChannels + 1 (for the time channel).
    minMaxScale = 1

    # Specify the saving parameters.
    sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]

    # -------------------------------------------------------------------- #
