
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
    specificEmotionModel = "specificEmotionModel"
    sharedEmotionModel = "sharedEmotionModel"
    trainingInformation = "trainingInformation"
    signalEncoderModel = "signalEncoderModel"
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
    timeWindows = [90, 120, 150, 180, 210, 240, 300]
    minNumExperiments = 55  # The minimum number of experiments that can be used in the model
    maxNumSignals = 128  # The maximum number of signals that can be used in the model (2**n)
    minMaxScale = 1  # The maximum value for the min-max scaling.

    # Specify the data interface parameter names.
    signalChannelNames = [timeChannel, signalChannel]

    # Specify the saving parameters.
    sharedModelWeights = ["sharedSignalEncoderModel", "sharedEmotionModel"]

    # -------------------------------------------------------------------- #
