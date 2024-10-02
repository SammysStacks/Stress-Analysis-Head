# General

# PyTorch
import torch
from torch import nn

# Import helper modules
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel
from .emotionModelHelpers.submodels.trainingInformation import trainingInformation


class emotionModelHead(nn.Module):
    def __init__(self, submodel, accelerator, finalDistributionLength, signalMinMaxScale, metadata, userInputParams,
                 timeWindows, emotionNames, activityNames, featureNames, numSubjects, datasetName, useFinalParams, debuggingResults=False):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.sequenceBounds = (timeWindows[0], timeWindows[-1])  # The minimum and maximum sequence length for the model.
        self.finalDistributionLength = finalDistributionLength  # The final length of the signal distribution.
        self.metadata = metadata  # The subject identifiers for the model (e.g., subjectIndex, datasetIndex, etc.)
        self.signalMinMaxScale = signalMinMaxScale  # The minimum and maximum values for the signals.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.device = accelerator.device  # The device the model is running on.
        self.useFinalParams = useFinalParams  # Whether to use the HPC parameters.
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.datasetName = datasetName  # The name of the dataset the model is training on.

        # Signal encoder parameters.
        self.signalEncoderWaveletType = userInputParams['signalEncoderWaveletType']  # The type of wavelet to use for signal encoding.
        self.numSigLiftedChannels = userInputParams['numSigLiftedChannels']     # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = userInputParams['numSigEncodingLayers']     # The number of layers during signal encoding.
        self.encodedSamplingFreq = userInputParams['encodedSamplingFreq']     # The number of transformer layers during signal encoding.
        self.latentQueryKeyDim = 4
        self.finalSignalDim = 1024

        # Emotion parameters.
        self.numInterpreterHeads = userInputParams['numInterpreterHeads']   # The number of ways to interpret a set of physiological signals.
        self.numBasicEmotions = userInputParams['numBasicEmotions']         # The number of basic emotions (basis states of emotions).

        # Tunable encoding parameters.
        self.numEncodedSignals = 1  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = 64  # The final length of the compressed signal after the autoencoder.
        # Feature parameters (code changes required if you change these!!!)
        self.numCommonSignals = 8    # The number of features from considering all the signals.
        self.numEmotionSignals = 8   # The number of common activity features to extract.
        self.numActivitySignals = 8  # The number of common activity features to extract.

        # Setup holder for the model's training information
        self.trainingInformation = trainingInformation()

        # ------------------------ Data Compression ------------------------ # 

        # The signal encoder model to find a common feature vector across all signals.
        self.specificSignalEncoderModel = specificSignalEncoderModel(
            latentQueryKeyDim=self.latentQueryKeyDim,
            finalSignalDim=self.finalSignalDim,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            numSigLiftedChannels=self.numSigLiftedChannels,
            numSigEncodingLayers=self.numSigEncodingLayers,
            encodedSamplingFreq=self.encodedSamplingFreq,
            waveletType=self.signalEncoderWaveletType,
            numEncodedSignals=self.numEncodedSignals,
            signalMinMaxScale=self.signalMinMaxScale,
            latentQueryKeyDim=self.latentQueryKeyDim,
            debuggingResults=self.debuggingResults,
            finalSignalDim=self.finalSignalDim,
            sequenceBounds=self.sequenceBounds,
            accelerator=self.accelerator,
        )

        self.specificEmotionModel = None
        self.sharedEmotionModel = None

        # -------------------- Final Emotion Prediction -------------------- #

        # self.specificEmotionModel = specificEmotionModel(
        #     numInterpreterHeads=self.numInterpreterHeads,
        #     numActivityFeatures=self.numCommonSignals,
        #     numCommonSignals=self.numCommonSignals,
        #     numBasicEmotions=self.numBasicEmotions,
        #     activityNames=self.activityNames,
        #     emotionNames=self.emotionNames,
        #     featureNames=self.featureNames,
        #     numSubjects=self.numSubjects,
        # )
        #
        # self.sharedEmotionModel = sharedEmotionModel(
        #     numActivityFeatures=self.numCommonSignals,
        #     numInterpreterHeads=self.numInterpreterHeads,
        #     numCommonSignals=self.numCommonSignals,
        #     numBasicEmotions=self.numBasicEmotions,
        #     numEncodedSignals=self.numEncodedSignals,
        #     compressedLength=self.compressedLength,
        # )

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, submodel, signalData, signalIdentifiers, metadata, trainingFlag=False):
        # decodeSignals: whether to decode the signals after encoding, which is used for the autoencoder loss.
        # trainingFlag: whether the model is training or testing.
        with self.accelerator.autocast():
            # Preprocess the data to ensure integrity.
            signalData, signalIdentifiers, metadata = (tensor.to(self.device) for tensor in (signalData, signalIdentifiers, metadata))
            batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
            assert numChannels == len(modelConstants.signalChannelNames)

            # Initialize default output tensors.
            basicEmotionProfile = torch.zeros((batchSize, numSignals, maxSequenceLength, numChannels), device=signalData.device)
            emotionProfile = torch.zeros((batchSize, numSignals, maxSequenceLength, numChannels), device=signalData.device)

            # ------------------- Learned Signal Compression ------------------- #

            print(signalData.size())
            print(signalData[0][0])

            # Interpolate the data to a fixed input size.
            interpolatedSignals = self.sharedSignalEncoderModel.learnedInterpolation(signalData=signalData)
            # interpolatedData: batchSize, numSignals, finalSignalDim

            print(interpolatedSignals.size())
            print(interpolatedSignals[0][0])

            # ---------------------- Emotion Model ---------------------- #

            if submodel == modelConstants.emotionPredictionModel:
                activityProfile, basicEmotionProfile, emotionProfile = self.emotionPrediction(signalData, metadata)

            return interpolatedSignals, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile
