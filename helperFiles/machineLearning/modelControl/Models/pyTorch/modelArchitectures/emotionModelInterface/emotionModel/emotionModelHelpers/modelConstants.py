
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
    specificEmotionModel = "specificEmotionModel"
    trainingInformation = "trainingInformation"
    signalEncoderModel = "signalEncoderModel"
    sharedEmotionModel = "sharedEmotionModel"
    signalMappingModel = "signalMappingModel"
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
    maxNumSignals = 128  # The maximum number of signals that can be used in the model (2**n)
    minMaxScale = 1

    # Specify the data interface parameter names.
    signalChannelNames = ["signal", "dTimeBack", "dTimeForward"]

    # Specify the saving parameters.
    sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]

    # -------------------------------------------------------------------- #
