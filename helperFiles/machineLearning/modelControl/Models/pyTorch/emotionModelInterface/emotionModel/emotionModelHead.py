# General
import time

# PyTorch
import torch
from matplotlib import pyplot as plt
from torch import nn

# Import helper modules
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel
from .emotionModelHelpers.submodels.trainingInformation import trainingInformation


class emotionModelHead(nn.Module):
    def __init__(self, submodel, metadata, userInputParams, emotionNames, activityNames, featureNames, numSubjects, datasetName, debuggingResults=False):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.datasetName = datasetName  # The name of the dataset the model is training on.
        self.metadata = metadata  # The subject identifiers for the model (e.g., subjectIndex, datasetIndex, etc.)

        # Operation parameters.
        self.operatorType = userInputParams['operatorType']  # The type of operator to use for the neural operator.

        # Signal encoder parameters.
        self.numSpecificEncodingLayers = userInputParams['numSpecificEncodingLayers']    # The number of layers in the dataset-specific signal encoding.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']  # The parameters for the neural operator.
        self.numMetaEncodingLayers = userInputParams['numMetaEncodingLayers']    # The number of layers in the shared signal encoding operator.
        self.latentQueryKeyDim = userInputParams['latentQueryKeyDim']     # The dimension of the latent query and key vectors.
        self.encodedDimension = userInputParams['encodedDimension']     # The dimension of the encoded signal.
        self.activationMethod = 'reversibleLinearSoftSign_2_0.9'  # The activation method to use for the neural operator.

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
        self.reversibleInterface = reversibleInterface()
        self.trainingInformation = trainingInformation()

        # ------------------------ Data Compression ------------------------ # 

        # The signal encoder model to find a common feature vector across all signals.
        self.specificSignalEncoderModel = specificSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numOperatorLayers=self.numSpecificEncodingLayers,
            activationMethod=self.activationMethod,
            sequenceLength=self.encodedDimension,
            numInputSignals=self.numSignals,
            operatorType=self.operatorType,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numOperatorLayers=self.numSpecificEncodingLayers,
            latentQueryKeyDim=self.latentQueryKeyDim,
            encodedDimension=self.encodedDimension,
            activationMethod=self.activationMethod,
            sequenceLength=self.encodedDimension,
            numInputSignals=self.numSignals,
            operatorType=self.operatorType,
        )

        self.specificEmotionModel = None
        self.sharedEmotionModel = None

        # -------------------- Final Emotion Prediction -------------------- #

        if submodel == modelConstants.emotionModel:
            self.specificEmotionModel = specificEmotionModel(
                numInterpreterHeads=self.numInterpreterHeads,
                numActivityFeatures=self.numCommonSignals,
                numCommonSignals=self.numCommonSignals,
                numBasicEmotions=self.numBasicEmotions,
                activityNames=self.activityNames,
                emotionNames=self.emotionNames,
                featureNames=self.featureNames,
                numSubjects=self.numSubjects,
            )

            self.sharedEmotionModel = sharedEmotionModel(
                numActivityFeatures=self.numCommonSignals,
                numInterpreterHeads=self.numInterpreterHeads,
                numCommonSignals=self.numCommonSignals,
                numBasicEmotions=self.numBasicEmotions,
                numEncodedSignals=self.numEncodedSignals,
                compressedLength=self.compressedLength,
            )

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, accelerator, submodel, signalData, signalIdentifiers, metadata, trainingFlag=False):
        # decodeSignals: whether to decode the signals after encoding, which is used for the autoencoder loss.
        # trainingFlag: whether the model is training or testing.
        # Preprocess the data to ensure integrity.
        signalData, signalIdentifiers, metadata = (tensor.to(signalData.device) for tensor in (signalData, signalIdentifiers, metadata))
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        reversibleInterface.changeDirections(forwardDirection=True)
        signalData = signalData.double()

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, numSignals, maxSequenceLength, numChannels), device=signalData.device)
        emotionProfile = torch.zeros((batchSize, numSignals, maxSequenceLength, numChannels), device=signalData.device)

        # ------------------- Estimated Physiological Profile ------------------- #

        # Interpolate the data to a fixed input size.
        interpolatedSignalData = self.sharedSignalEncoderModel.learnedInterpolation(signalData=signalData)
        interpolatedSignalData = interpolatedSignalData*100
        # interpolatedData: batchSize, numSignals, encodedDimension
        print(0, interpolatedSignalData[0][0][0:10])

        # Calculate the estimated physiological profile given each signal.
        metaLearningData = self.specificSignalEncoderModel.signalSpecificInterface(signalData=interpolatedSignalData, initialModel=True)  # Reversible signal-specific layers.
        metaLearningData = self.sharedSignalEncoderModel.sharedLearning(signalData=metaLearningData)  # Reversible meta-learning layers.
        metaLearningData = self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningData, initialModel=False)  # Reversible signal-specific layers.
        # metaLearningData: batchSize, numSignals, encodedDimension

        # Finalize the physiological profile.
        physiologicalProfile = metaLearningData.mean(dim=1)
        # interpolatedData: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Remap the signal data to the estimated physiological profile.
        remappedSignalData = physiologicalProfile.unsqueeze(1).repeat(1, numSignals, 1)
        reversibleInterface.changeDirections(forwardDirection=False)

        # Calculate the estimated physiological profile given each signal.
        metaLearningData = self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningData, initialModel=False)  # Reversible signal-specific layers.
        metaLearningData = self.sharedSignalEncoderModel.sharedLearning(signalData=metaLearningData)  # Reversible meta-learning layers.
        reconstructedInterpolatedData = self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningData, initialModel=True)  # Reversible signal-specific layers.
        print(0, reconstructedInterpolatedData[0][0][0:10])
        # metaLearningData: batchSize, numSignals, encodedDimension

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(interpolatedSignalData[0][0].detach().cpu().numpy(), 'k', linewidth=2, label='Interpolated Signal')
        plt.plot(reconstructedInterpolatedData[0][0].detach().cpu().numpy(), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
        plt.plot(physiologicalProfile[0].detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Physiological Profile', alpha=0.5)
        plt.legend()
        plt.show()
        exit()

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            activityProfile, basicEmotionProfile, emotionProfile = self.emotionPrediction(signalData, metadata)

        return interpolatedSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile
