import random
import time

import torch
from matplotlib import pyplot as plt
from torch import nn

from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
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
    def __init__(self, submodel, metadata, userInputParams, emotionNames, activityNames, featureNames, numSubjects, datasetName, numExperiments):
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
        self.encodedDimension = userInputParams['encodedDimension']  # The dimension of the encoded signal.
        self.learningProtocol = userInputParams['learningProtocol']  # The learning protocol for the model.
        self.fourierDimension = int(self.encodedDimension/2 + 1)  # The dimension of the fourier signal.
        self.operatorType = userInputParams['operatorType']  # The type of operator to use for the neural operator.
        self.debugging = True

        # Signal encoder parameters.
        self.numSpecificEncodingLayers = userInputParams['numSpecificEncodingLayers']  # The number of layers in the dataset-specific signal encoding.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']  # The parameters for the neural operator.
        self.numMetaEncodingLayers = userInputParams['numMetaEncodingLayers']  # The number of layers in the shared signal encoding operator.
        self.numLiftingLayers = 2  # The number of lifting layers to use in the signal encoder.

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
            learningProtocol=self.learningProtocol,
            fourierDimension=self.fourierDimension,
            numLiftingLayers=self.numLiftingLayers,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
            numSignals=self.numSignals,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numOperatorLayers=self.numMetaEncodingLayers,
            encodedDimension=self.encodedDimension,
            activationMethod=self.activationMethod,
            learningProtocol=self.learningProtocol,
            fourierDimension=self.fourierDimension,
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
                numInterpreterHeads=self.numInterpreterHeads,
                numActivityFeatures=self.numCommonSignals,
                numEncodedSignals=self.numEncodedSignals,
                numCommonSignals=self.numCommonSignals,
                numBasicEmotions=self.numBasicEmotions,
                compressedLength=self.compressedLength,
            )

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, trainingFlag=False):
        # decodeSignals: whether to decode the signals after encoding, which is used for the autoencoder loss.
        # trainingFlag: whether the model is training or testing.
        # Preprocess the data to ensure integrity.
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        reversibleInterface.changeDirections(forwardDirection=True)

        # Ensure high numerical precision for the data.
        signalIdentifiers = signalIdentifiers.int()
        signalData = signalData.double()
        metadata = metadata.double()

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
        generalEncodingLoss = torch.zeros(batchSize, device=device)

        # ------------------- Estimated Physiological Profile ------------------- #

        # Unpack the incoming data.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        # datapoints and timepoints: [batchSize, numSignals, maxSequenceLength)
        # timepoints: [further away from survey (300) -> closest to survey (0)]
        # batchInds: batchSize

        # Check which points were missing in the data.
        missingDataMask = torch.as_tensor((datapoints == 0) & (timepoints == 0), device=datapoints.device)
        validSignalMask = ~torch.all(missingDataMask, dim=-1).unsqueeze(-1)
        numValidSignals = validSignalMask.sum(dim=1).float()
        # missingDataMask: batchSize, numSignals, maxSequenceLength
        # validSignalMask: batchSize, numSignals, 1
        # numValidSignals: batchSize

        # Get the estimated physiological profiles.
        physiologicalProfile = self.specificSignalEncoderModel.getPhysiologicalProfileEstimation(batchInds, trainingFlag=trainingFlag)
        reversibleInterface.changeDirections(forwardDirection=False)
        # physiologicalProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Remap the signal data to the estimated physiological profile.
        fourierMagnitudeData, fourierPhaseData = self.sharedSignalEncoderModel.forwardFFT(physiologicalProfile)
        fourierMagnitudeData = validSignalMask * fourierMagnitudeData.unsqueeze(1).repeat(1, numSignals, 1)
        fourierPhaseData = validSignalMask * fourierPhaseData.unsqueeze(1).repeat(1, numSignals, 1)
        # fourierMagnitudeData and fourierPhaseData: batchSize, numSignals, fourierDimension
        # physiologicalFourierData: batchSize, 2*numSignals, fourierDimension

        # Combine the magnitude and phase data.
        physiologicalFourierData = torch.cat(tensors=(fourierMagnitudeData, fourierPhaseData), dim=1)
        validFourierMask = validSignalMask.repeat(repeats=(1, 2, 1))
        # physiologicalFourierData: batchSize, 2*numSignals, fourierDimension
        # validFourierMask: batchSize, 2*numSignals

        # Calculate the estimated physiological profile given each signal.
        metaLearningDataR2 = validFourierMask * self.specificSignalEncoderModel.signalSpecificInterface(signalData=physiologicalFourierData, initialModel=False)  # Reversible signal-specific layers.
        metaLearningDataR1 = validFourierMask * self.sharedSignalEncoderModel.sharedLearning(signalData=metaLearningDataR2)  # Reversible meta-learning layers.
        fourierData = validFourierMask * self.specificSignalEncoderModel.signalSpecificInterface(signalData=metaLearningDataR1, initialModel=True)  # Reversible signal-specific layers.
        # metaLearningData: batchSize, numSignals, fourierDimension

        # Reconstruct the signal data from the Fourier data.
        fourierMagnitudeData, fourierPhaseData = fourierData[:, :numSignals], fourierData[:, numSignals:]
        reconstructedSignalData = validSignalMask * self.sharedSignalEncoderModel.backwardFFT(fourierMagnitudeData, fourierPhaseData)
        # fourierMagnitudeData and fourierPhaseData: batchSize, numSignals, fourierDimension
        # reconstructedSignalData: batchSize, numSignals, encodedDimension

        if self.debugging and random.random() < 0.05:
            # Optionally, plot the physiological profile for visual comparison
            physiologicalTimes = self.sharedSignalEncoderModel.pseudoEncodedTimes.detach().cpu().numpy()
            plt.plot(physiologicalTimes, physiologicalProfile[0].detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Physiological Profile', alpha=0.5)
            plt.legend()
            plt.show()

            # Optionally, plot the original and reconstructed signals for visual comparison
            plt.plot(timepoints[0, 0, ~missingDataMask[0][0]].detach().cpu().numpy(), datapoints[0, 0, ~missingDataMask[0][0]].detach().cpu().numpy(), 'ok', markersize=3, label='Initial Signal', alpha=0.5)
            plt.plot(physiologicalTimes, reconstructedSignalData[0, 0, :].detach().cpu().numpy(), 'tab:red', linewidth=1, label='Reconstructed Signal')
            plt.legend()
            plt.show()

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            activityProfile, basicEmotionProfile, emotionProfile = self.emotionPrediction(signalData, metadata)

        return missingDataMask, reconstructedSignalData, generalEncodingLoss, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, trainingFlag):
        with torch.no_grad():
            # Preallocate the output tensors.
            batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
            reconstructedSignalData = torch.zeros((batchSize, numSignals, self.encodedDimension), device=device)
            basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
            emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
            missingDataMask = torch.zeros((batchSize, numSignals, maxSequenceLength), device=device)
            physiologicalProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
            activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
            testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
            generalEncodingLoss = torch.zeros(batchSize, device=device)
            startBatchInd = 0

            while startBatchInd + testingBatchSize < batchSize:
                endBatchInd = startBatchInd + testingBatchSize

                # Perform a full pass of the model.
                missingDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], generalEncodingLoss[startBatchInd:endBatchInd], physiologicalProfile[startBatchInd:endBatchInd], activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                    = self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd], metadata=metadata[startBatchInd:endBatchInd], device=device, trainingFlag=trainingFlag)

                # Update the batch index.
                startBatchInd = endBatchInd

        return missingDataMask, reconstructedSignalData, generalEncodingLoss, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile
