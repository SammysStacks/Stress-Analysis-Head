# General
import random

# PyTorch
import torch
from matplotlib import pyplot as plt
from torch import nn

# Import helper modules
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.modelParameters import modelParameters
from .emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from .emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel
from .emotionModelHelpers.submodels.trainingInformation import trainingInformation


class emotionModelHead(nn.Module):
    def __init__(self, submodel, metadata, userInputParams, emotionNames, activityNames, featureNames, numSubjects, datasetName):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.datasetName = datasetName  # The name of the dataset the model is training on.
        self.metadata = metadata  # The subject identifiers for the model (e.g., subjectIndex, datasetIndex, etc.)

        # General parameters.
        self.activationMethod = emotionModelWeights.getActivationType()  # The activation method to use for the neural operator.
        self.operatorType = userInputParams['operatorType']  # The type of operator to use for the neural operator.
        self.debugging = True

        # Signal encoder parameters.
        self.numSpecificEncodingLayers = userInputParams['numSpecificEncodingLayers']  # The number of layers in the dataset-specific signal encoding.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']  # The parameters for the neural operator.
        self.numMetaEncodingLayers = userInputParams['numMetaEncodingLayers']  # The number of layers in the shared signal encoding operator.
        self.encodedDimension = userInputParams['encodedDimension']  # The dimension of the encoded signal.

        # Emotion parameters.
        self.numInterpreterHeads = userInputParams['numInterpreterHeads']  # The number of ways to interpret a set of physiological signals.
        self.numBasicEmotions = userInputParams['numBasicEmotions']  # The number of basic emotions (basis states of emotions).

        # Tunable encoding parameters.
        self.numEncodedSignals = 1  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = 64  # The final length of the compressed signal after the autoencoder.
        # Feature parameters (code changes required if you change these!!!)
        self.numCommonSignals = 8  # The number of features from considering all the signals.
        self.numEmotionSignals = 8  # The number of common activity features to extract.
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
            encodedDimension=self.encodedDimension,
            numInputSignals=self.numSignals,
            operatorType=self.operatorType,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numOperatorLayers=self.numMetaEncodingLayers,
            encodedDimension=self.encodedDimension,
            activationMethod=self.activationMethod,
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

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, fullDataPass=False):
        # decodeSignals: whether to decode the signals after encoding, which is used for the autoencoder loss.
        # trainingFlag: whether the model is training or testing.
        # Preprocess the data to ensure integrity.
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        reversibleInterface.changeDirections(forwardDirection=True)

        # Ensure high numerical precision for the data.
        signalData = signalData.round(decimals=6).double()
        signalIdentifiers = signalIdentifiers.double()
        metadata = metadata.double()

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        reconstructedInterpolatedData = torch.zeros((batchSize, numSignals, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
        finalManifoldProjectionLoss = torch.zeros(batchSize, device=device)

        # ------------------- Estimated Physiological Profile ------------------- #

        # Interpolate the data to a fixed input size.
        interpolatedSignalData, missingDataMask = self.specificSignalEncoderModel.learnedInterpolation(signalData=signalData)
        # missingDataMask: batchSize, numSignals, maxSequenceLength
        # interpolatedData: batchSize, numSignals, encodedDimension

        # Calculate the estimated physiological profile given each signal.
        metaLearningDataF1 = self.specificSignalEncoderModel.signalSpecificInterface(signalData=interpolatedSignalData, initialModel=True)  # Reversible signal-specific layers.
        metaLearningDataF2 = self.sharedSignalEncoderModel.sharedLearning(signalData=metaLearningDataF1)  # Reversible meta-learning layers.
        metaLearningDataF3 = self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningDataF2, initialModel=False)  # Reversible signal-specific layers.
        # metaLearningData: batchSize, numSignals, encodedDimension

        # Remove the influence of missing data from the path.
        missingSignalMask = (missingDataMask == 0).all(dim=-1)
        interpolatedSignalData = missingSignalMask*interpolatedSignalData
        metaLearningDataF1 = missingSignalMask*metaLearningDataF1
        metaLearningDataF2 = missingSignalMask*metaLearningDataF2
        metaLearningDataF3 = missingSignalMask*metaLearningDataF3
        # missingDataMask: batchSize, numSignals

        # Finalize the physiological profile.
        numValidSignals = numSignals - missingSignalMask.sum(dim=1)  # Dim: batchSize
        physiologicalProfile = metaLearningDataF3.sum(dim=1) / numValidSignals.unsqueeze(-1)
        # interpolatedData: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Remap the signal data to the estimated physiological profile.
        remappedSignalData = physiologicalProfile.unsqueeze(1).repeat(1, numSignals, 1)
        reversibleInterface.changeDirections(forwardDirection=False)

        if fullDataPass:
            # Calculate the estimated physiological profile given each signal.
            metaLearningDataR2 = self.specificSignalEncoderModel.signalSpecificInterface(signalData=remappedSignalData, initialModel=False)  # Reversible signal-specific layers.
            metaLearningDataR1 = self.sharedSignalEncoderModel.sharedLearning(signalData=metaLearningDataR2)  # Reversible meta-learning layers.
            reconstructedInterpolatedData = self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningDataR1, initialModel=True)  # Reversible signal-specific layers.
            # metaLearningData: batchSize, numSignals, encodedDimension

            # Remove the influence of missing data from the path.
            reconstructedInterpolatedData = missingSignalMask * reconstructedInterpolatedData
            metaLearningDataR2 = missingSignalMask * metaLearningDataR2
            metaLearningDataR1 = missingSignalMask * metaLearningDataR1
            remappedSignalData = missingSignalMask * remappedSignalData
            # metaLearningData: batchSize, numSignals, encodedDimension

            # Loss calculation for the signal encoder.
            finalManifoldProjectionLoss = finalManifoldProjectionLoss + (metaLearningDataF3 - remappedSignalData).pow(2).mean(dim=2).sum(dim=1) / numValidSignals
            finalManifoldProjectionLoss = finalManifoldProjectionLoss + (metaLearningDataF2 - metaLearningDataR2).pow(2).mean(dim=2).sum(dim=1) / numValidSignals
            finalManifoldProjectionLoss = finalManifoldProjectionLoss + (metaLearningDataF1 - metaLearningDataR1).pow(2).mean(dim=2).sum(dim=1) / numValidSignals

            if self.debugging and True:
                physiologicalTimes = self.specificSignalEncoderModel.ebbinghausInterpolation.pseudoEncodedTimes.detach().cpu().numpy()
                # Optionally, plot the original and reconstructed signals for visual comparison
                plt.plot(signalData[0][0][:, 0].detach().cpu().numpy(), signalData[0][0][:, 1].detach().cpu().numpy(), 'k', linewidth=2, label='Initial Signal', alpha=0.5)
                plt.plot(physiologicalTimes, interpolatedSignalData[0][0].detach().cpu().numpy(), 'tab:red', linewidth=1, label='Interpolated Signal')
                plt.legend()
                plt.show()

                plt.plot(physiologicalTimes, interpolatedSignalData[0][0].detach().cpu().numpy(), 'k', linewidth=2, label='Interpolated Signal')
                plt.plot(physiologicalTimes, reconstructedInterpolatedData[0][0].detach().cpu().numpy(), 'tab:red', linewidth=1.5, label='Reconstructed Signal')
                plt.plot(physiologicalTimes, physiologicalProfile[0].detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Physiological Profile', alpha=0.5)
                plt.legend()
                plt.show()

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            activityProfile, basicEmotionProfile, emotionProfile = self.emotionPrediction(signalData, metadata)

        return missingDataMask, interpolatedSignalData, finalManifoldProjectionLoss, reconstructedInterpolatedData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, fullDataPass=True):
        with torch.no_grad():
            # Preallocate the output tensors.
            batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
            interpolatedSignalData, reconstructedInterpolatedData = (torch.zeros((batchSize, numSignals, self.encodedDimension), device=device) for _ in range(2))
            basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
            emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
            missingDataMask = torch.zeros((batchSize, numSignals, maxSequenceLength), device=device)
            physiologicalProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
            activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
            testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
            finalManifoldProjectionLoss = torch.zeros(batchSize, device=device)
            startBatchInd = 0

            while startBatchInd + testingBatchSize < batchSize:
                endBatchInd = startBatchInd + testingBatchSize

                # Perform a full pass of the model.
                missingDataMask[startBatchInd:endBatchInd], interpolatedSignalData[startBatchInd:endBatchInd], finalManifoldProjectionLoss[startBatchInd:endBatchInd], reconstructedInterpolatedData[startBatchInd:endBatchInd], physiologicalProfile[startBatchInd:endBatchInd], activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                    = self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd], metadata=metadata[startBatchInd:endBatchInd], device=device, fullDataPass=fullDataPass)

                # Update the batch index.
                startBatchInd = endBatchInd

        return missingDataMask, interpolatedSignalData, finalManifoldProjectionLoss, reconstructedInterpolatedData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile
