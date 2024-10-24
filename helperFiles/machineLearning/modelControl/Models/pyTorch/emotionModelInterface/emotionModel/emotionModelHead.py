import random

import torch
from matplotlib import pyplot as plt
from torch import nn

from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
# Import helper modules
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.modelParameters import modelParameters
from .emotionModelHelpers.submodels.inferenceModel import inferenceModel
from .emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from .emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights
from .emotionModelHelpers.submodels.sharedActivityModel import sharedActivityModel
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
from .emotionModelHelpers.submodels.specificActivityModel import specificActivityModel
# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel


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
        self.irreversibleLearningProtocol = userInputParams['irreversibleLearningProtocol']  # The learning protocol for the model.
        self.reversibleLearningProtocol = userInputParams['reversibleLearningProtocol']  # The learning protocol for the model.
        self.activationMethod = emotionModelWeights.getActivationType()  # The activation method to use for the neural operator.
        self.encodedDimension = userInputParams['encodedDimension']  # The dimension of the encoded signal.
        self.fourierDimension = int(self.encodedDimension / 2 + 1)  # The dimension of the fourier signal.
        self.numModelLayers = userInputParams['numModelLayers']  # The number of layers in the model.
        self.operatorType = userInputParams['operatorType']  # The type of operator to use for the neural operator.
        self.goldenRatio = userInputParams['goldenRatio']  # The number of shared layers per specific layer.
        self.debugging = True

        # Signal encoder parameters.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']  # The parameters for the neural operator.
        self.numLiftingLayersSignalEncoder = 2  # The number of lifting layers to use: real, imaginary.

        # Emotion parameters.
        self.numActivityChannels = userInputParams['numActivityChannels']  # The number of activity channels to predict.
        self.numBasicEmotions = userInputParams['numBasicEmotions']  # The number of basic emotions (basis states of emotions).

        # Setup holder for the model's training information
        self.reversibleInterface = reversibleInterface()

        # ------------------------ Data Compression ------------------------ #

        # Inference interface for the model.
        self.inferenceModel = inferenceModel(encodedDimension=self.encodedDimension)

        # The signal encoder model to find a common feature vector across all signals.
        self.specificSignalEncoderModel = specificSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numLiftingLayers=self.numLiftingLayersSignalEncoder,
            learningProtocol=self.reversibleLearningProtocol,
            activationMethod=self.activationMethod,
            encodedDimension=self.encodedDimension,
            fourierDimension=self.fourierDimension,
            numModelLayers=self.numModelLayers,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
            goldenRatio=self.goldenRatio,
            numSignals=self.numSignals,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            activationMethod=self.activationMethod,
            fourierDimension=self.fourierDimension,
            numModelLayers=self.numModelLayers,
            operatorType=self.operatorType,
            goldenRatio=self.goldenRatio,
        )

        # -------------------- Final Emotion Prediction -------------------- #

        if submodel == modelConstants.emotionModel:
            self.specificEmotionModel = specificEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                activationMethod=self.activationMethod,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numModelLayers,
                operatorType=self.operatorType,
                goldenRatio=self.goldenRatio,
                numEmotions=self.numEmotions,
                numSubjects=self.numSubjects,
            )

            self.sharedEmotionModel = sharedEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                activationMethod=self.activationMethod,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numModelLayers,
                operatorType=self.operatorType,
            )

            self.specificActivityModel = specificActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                activationMethod=self.activationMethod,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numModelLayers,
                numActivities=self.numActivities,
                operatorType=self.operatorType,
                goldenRatio=self.goldenRatio,
            )

            self.sharedActivityModel = sharedActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                activationMethod=self.activationMethod,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numModelLayers,
                operatorType=self.operatorType,
            )

    # ------------------------- Full Forward Calls ------------------------- #

    def addNewLayer(self):
        # Adjust the model architecture.
        if self.numModelLayers % self.goldenRatio == 0: self.specificSignalEncoderModel.addLayer()
        self.sharedSignalEncoderModel.addLayer()
        self.numModelLayers += 1

        # Analyze the new architecture.
        numSpecificLayers = len(self.specificSignalEncoderModel.neuralLayers)
        numSharedLayers = len(self.sharedSignalEncoderModel.neuralLayers)

        # Inform the user of the model changes.
        print(f"numModelLayers: {self.numModelLayers}, Specific Layers: {len(self.specificSignalEncoderModel.neuralLayers)}, Shared Layers: {len(self.sharedSignalEncoderModel.neuralLayers)}")
        assert self.numModelLayers == numSharedLayers, f"The number of layers in the shared model ({numSharedLayers}) does not match the number of layers in the model ({self.numModelLayers})."
        if self.numModelLayers % self.goldenRatio == 0 and self.numModelLayers != 0: assert numSpecificLayers == 1 + self.numModelLayers // self.goldenRatio, f"The number of layers in the specific model ({numSpecificLayers}) does not match the number of layers in the model ({self.numModelLayers})."

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, inferenceTraining=False):
        # decodeSignals: whether to decode the signals after encoding, which is used for the autoencoder loss.
        # trainingFlag: whether the model is training or testing.
        # Preprocess the data to ensure integrity.
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.double()
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)

        # ------------------- Organize the Incoming Data ------------------- #

        # Unpack the incoming data.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        # datapoints and timepoints: [batchSize, numSignals, maxSequenceLength)
        # timepoints: [further away from survey (300) -> closest to survey (0)]

        # Check which points were missing in the data.
        missingDataMask = torch.as_tensor((datapoints == 0) & (timepoints == 0), device=device, dtype=torch.bool)
        validSignalMask = ~torch.all(missingDataMask, dim=-1).unsqueeze(-1)
        # missingDataMask: batchSize, numSignals, maxSequenceLength
        # validSignalMask: batchSize, numSignals, 1

        # ------------------- Estimated Physiological Profile ------------------- #

        # Get the estimated physiological profiles.
        if inferenceTraining: physiologicalProfile = self.inferenceModel.getCurrentPhysiologicalProfile(batchSize)
        else: physiologicalProfile = self.specificSignalEncoderModel.getCurrentPhysiologicalProfile(batchInds)
        # physiologicalProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Remap the signal data to the estimated physiological profile.
        realFourierData, imaginaryFourierData = self.sharedSignalEncoderModel.forwardFFT(physiologicalProfile)
        imaginaryFourierData = validSignalMask * imaginaryFourierData.unsqueeze(1).repeat(1, numSignals, 1)
        realFourierData = validSignalMask * realFourierData.unsqueeze(1).repeat(1, numSignals, 1)
        # realFourierData and imaginaryFourierData: batchSize, numSignals, fourierDimension
        # physiologicalFourierData: batchSize, 2*numSignals, fourierDimension

        # Combine the magnitude and phase data.
        physiologicalFourierData = torch.cat(tensors=(realFourierData, imaginaryFourierData), dim=1)
        validFourierMask = validSignalMask.repeat(repeats=(1, 2, 1))
        # physiologicalFourierData: batchSize, 2*numSignals, fourierDimension
        # validFourierMask: batchSize, 2*numSignals

        # Perform the backward pass: physiologically -> signal data.
        reversibleInterface.changeDirections(forwardDirection=False)
        metaLearningData = self.coreModelPass(validFourierMask, physiologicalFourierData, specificModel=self.specificSignalEncoderModel, sharedModel=self.sharedSignalEncoderModel)
        # metaLearningData: batchSize, numSignals, fourierDimension

        # Reconstruct the signal data from the Fourier data.
        realFourierData, imaginaryFourierData = metaLearningData[:, :numSignals], metaLearningData[:, numSignals:]
        reconstructedSignalData = validSignalMask * self.sharedSignalEncoderModel.backwardFFT(realFourierData, imaginaryFourierData)
        # realFourierData and imaginaryFourierData: batchSize, numSignals, fourierDimension
        # reconstructedSignalData: batchSize, numSignals, encodedDimension

        # Visualize the data transformations within signal encoding.
        if not inferenceTraining and random.random() < 0.01: self.visualizeSignalEncoding(physiologicalProfile, reconstructedSignalData, timepoints, datapoints, missingDataMask)

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            # Get the subject-specific indices.
            subjectInds = emotionDataInterface.getMetaDataChannel(metadata, channelName=modelConstants.subjectIndexMD)  # Dim: batchSize

            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            basicEmotionProfile = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, self.numEmotions*self.numBasicEmotions, 1))
            basicEmotionProfile = self.coreModelPass(dataMask=1, metaLearningData=basicEmotionProfile, specificModel=self.specificEmotionModel, sharedModel=self.sharedEmotionModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

            # Reconstruct the emotion data.
            basicEmotionProfile = basicEmotionProfile.view(batchSize, self.numEmotions, self.numBasicEmotions, self.encodedDimension)
            emotionProfile = self.specificEmotionModel.calculateEmotionProfile(basicEmotionProfile, subjectInds)

        # ------------------- Learned Activity Mapping ------------------- #

            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            activityProfile = self.coreModelPass(dataMask=1, metaLearningData=physiologicalProfile, specificModel=self.specificActivityModel, sharedModel=self.sharedActivityModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

        # --------------------------------------------------------------- #

        return missingDataMask, reconstructedSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    @staticmethod
    def coreModelPass(dataMask, metaLearningData, specificModel, sharedModel):
        specificLayerCounter = 0

        # For each layer in the model.
        for layerInd in range(specificModel.numModelLayers):
            # Calculate the estimated physiological profile given each signal.
            if layerInd % specificModel.goldenRatio == 0: metaLearningData = dataMask * specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData); specificLayerCounter += 1  # Reversible signal-specific layers.
            metaLearningData = dataMask * sharedModel.learningInterface(layerInd=layerInd, signalData=metaLearningData)  # Reversible meta-learning layers.
        metaLearningData = dataMask * specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData)  # Reversible signal-specific layers.
        assert specificLayerCounter + 1 == len(specificModel.neuralLayers), f"The specific layer counter ({specificLayerCounter}) does not match the number of specific layers ({len(specificModel.neuralLayers)})."
        # metaLearningData: batchSize, numSignals, finalDimension

        return metaLearningData

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, inferenceTraining):
        with torch.no_grad():
            # Preallocate the output tensors.
            numExperiments, numSignals, maxSequenceLength, numChannels = signalData.size()
            basicEmotionProfile = torch.zeros((numExperiments, self.numBasicEmotions, self.encodedDimension), device=device, dtype=torch.float64)
            reconstructedSignalData = torch.zeros((numExperiments, numSignals, self.encodedDimension), device=device, dtype=torch.float64)
            emotionProfile = torch.zeros((numExperiments, self.numEmotions, self.encodedDimension), device=device, dtype=torch.float64)
            missingDataMask = torch.zeros((numExperiments, numSignals, maxSequenceLength), device=device, dtype=torch.bool)
            physiologicalProfile = torch.zeros((numExperiments, self.encodedDimension), device=device, dtype=torch.float64)
            activityProfile = torch.zeros((numExperiments, self.encodedDimension), device=device, dtype=torch.float64)
            testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
            startBatchInd = 0

            while startBatchInd < numExperiments:
                endBatchInd = startBatchInd + testingBatchSize

                # Perform a full pass of the model.
                missingDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], physiologicalProfile[startBatchInd:endBatchInd], \
                    activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                    = self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd],
                                   metadata=metadata[startBatchInd:endBatchInd], device=device, inferenceTraining=inferenceTraining)

                # Update the batch index.
                startBatchInd = endBatchInd

        return missingDataMask, reconstructedSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    def visualizeSignalEncoding(self, physiologicalProfile, reconstructedSignalData, timepoints, datapoints, missingDataMask):
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
