import random

import torch
from matplotlib import pyplot as plt
from torch import nn

from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.modelParameters import modelParameters
from .emotionModelHelpers.submodels.inferenceModel import inferenceModel
from .emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from .emotionModelHelpers.submodels.sharedActivityModel import sharedActivityModel
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
from .emotionModelHelpers.submodels.specificActivityModel import specificActivityModel
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel


class emotionModelHead(nn.Module):
    def __init__(self, submodel, userInputParams, emotionNames, activityNames, featureNames, numSubjects, datasetName, numExperiments):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.datasetName = datasetName  # The name of the dataset the model is training on.

        # General parameters.
        self.encodedDimension = userInputParams['encodedDimension']  # The dimension of the encoded signal.
        self.operatorType = userInputParams['operatorType']  # The type of operator to use for the neural operator.
        self.goldenRatio = userInputParams['goldenRatio']  # The number of shared layers per specific layer.
        self.debugging = True

        # Signal encoder parameters.
        self.reversibleLearningProtocol = userInputParams['reversibleLearningProtocol']   # The learning protocol for the model.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']   # The parameters for the neural operator.
        self.numSignalEncoderLayers = 0  # The number of layers in the model. Added downstream.

        # Emotion and activity parameters.
        self.irreversibleLearningProtocol = userInputParams['irreversibleLearningProtocol']  # The learning protocol for the model.
        self.numActivityModelLayers = userInputParams['numActivityModelLayers']  # The number of basic emotions (basis states of emotions).
        self.numEmotionModelLayers = userInputParams['numEmotionModelLayers']  # The number of basic emotions (basis states of emotions).
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
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            featureNames=self.featureNames,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
            goldenRatio=self.goldenRatio,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            operatorType=self.operatorType,
            goldenRatio=self.goldenRatio,
        )

        # Construct the model weights.
        for _ in range(userInputParams['numSignalEncoderLayers']): self.addNewSignalEncoderLayer()
        self.specificSignalEncoderModel.addLayer()  # Add the final layer to the specific model.

        # -------------------- Final Emotion Prediction -------------------- #

        if submodel == modelConstants.emotionModel:
            self.specificEmotionModel = specificEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numEmotionModelLayers,
                operatorType=self.operatorType,
                goldenRatio=self.goldenRatio,
                numEmotions=self.numEmotions,
                numSubjects=self.numSubjects,

            )

            self.sharedEmotionModel = sharedEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numEmotionModelLayers,
                operatorType=self.operatorType,
                numEmotions=self.numEmotions,
            )

            self.specificActivityModel = specificActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numActivityModelLayers,
                numActivities=self.numActivities,
                operatorType=self.operatorType,
                goldenRatio=self.goldenRatio,
            )

            self.sharedActivityModel = sharedActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numActivityModelLayers,
                operatorType=self.operatorType,
            )

    # ------------------------- Full Forward Calls ------------------------- #

    def addNewSignalEncoderLayer(self):
        # Adjust the model architecture.
        if self.numSignalEncoderLayers % self.goldenRatio == 0: self.specificSignalEncoderModel.addLayer()
        self.sharedSignalEncoderModel.addLayer()
        self.numSignalEncoderLayers += 1

        # Analyze the new architecture.
        numSpecificLayers = len(self.specificSignalEncoderModel.neuralLayers)
        numSharedLayers = len(self.sharedSignalEncoderModel.neuralLayers)

        # Assert the validity of the model architecture.
        assert self.numSignalEncoderLayers == numSharedLayers, f"The number of layers in the shared model ({numSharedLayers}) does not match the number of layers in the model ({self.numSignalEncoderLayers})."
        if self.numSignalEncoderLayers % self.goldenRatio == 0 and self.numSignalEncoderLayers != 0: assert numSpecificLayers == self.numSignalEncoderLayers // self.goldenRatio, f"The number of layers in the specific model ({numSpecificLayers}) does not match the number of layers in the model ({self.numSignalEncoderLayers})."

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, inferenceTraining=False):
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.int()
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        # timepoints: [further away from survey (300) -> closest to survey (0)]
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        # signalIdentifiers: [batchSize, numSignals, numSignalIdentifiers]
        # metadata: [batchSize, numMetadata]

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)

        # ------------------- Organize the Incoming Data ------------------- #

        # Check which points were missing in the data.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        validDataMask = emotionDataInterface.getValidDataMask(signalData)
        # missingDataMask: batchSize, numSignals, maxSequenceLength

        # ------------------- Estimated Physiological Profile ------------------- #

        # Get the estimated physiological profiles.
        if inferenceTraining: physiologicalProfile = self.inferenceModel.getCurrentPhysiologicalProfile(batchInds)
        else: physiologicalProfile = self.specificSignalEncoderModel.profileModel.getCurrentPhysiologicalProfile(batchInds)
        physiologicalProfile = self.specificSignalEncoderModel.smoothingFilter(physiologicalProfile.unsqueeze(1), kernelSize=3).squeeze(1)
        # physiologicalProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Perform the backward pass: physiologically -> signal data.
        reversibleInterface.changeDirections(forwardDirection=False)
        resampledSignalData = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, numSignals, 1))
        resampledSignalData = self.coreModelPass(self.numSignalEncoderLayers, resampledSignalData, specificModel=self.specificSignalEncoderModel, sharedModel=self.sharedSignalEncoderModel)
        # resampledSignalData: batchSize, numSignals, encodedDimension

        # Resample the signal data.
        reconstructedSignalData = self.interpolateData(signalData, resampledSignalData)
        # reconstructedSignalData: batchSize, numSignals, maxSequenceLength

        # Visualize the data transformations within signal encoding.
        if submodel == modelConstants.signalEncoderModel and not inferenceTraining and random.random() < 0.01:
            with torch.no_grad(): self.visualizeSignalEncoding(physiologicalProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask)

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            basicEmotionProfile = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, self.numBasicEmotions, 1))
            basicEmotionProfile = self.coreModelPass(self.numEmotionModelLayers, metaLearningData=basicEmotionProfile, specificModel=self.specificEmotionModel, sharedModel=self.sharedEmotionModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

            # Reconstruct the emotion data.
            basicEmotionProfile = basicEmotionProfile.repeat(repeats=(1, self.numEmotions, 1, 1))
            subjectInds = emotionDataInterface.getMetaDataChannel(metadata, channelName=modelConstants.subjectIndexMD)  # Dim: batchSize
            emotionProfile = self.specificEmotionModel.calculateEmotionProfile(basicEmotionProfile, subjectInds)

        # ------------------- Learned Activity Mapping ------------------- #

            # Perform the backward pass: physiologically -> activity data.
            reversibleInterface.changeDirections(forwardDirection=False)
            resampledActivityData = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, self.numActivityChannels, 1))
            activityProfile = self.coreModelPass(self.numActivityModelLayers, metaLearningData=resampledActivityData, specificModel=self.specificActivityModel, sharedModel=self.sharedActivityModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

        # --------------------------------------------------------------- #

        return validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    @staticmethod
    def coreModelPass(numModelLayers, metaLearningData, specificModel, sharedModel):
        specificLayerCounter = 0

        # For each layer in the model.
        for layerInd in range(numModelLayers):
            # Calculate the estimated physiological profile given each signal.
            if layerInd % specificModel.goldenRatio == 0: metaLearningData = specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData); specificLayerCounter += 1
            metaLearningData = sharedModel.learningInterface(layerInd=layerInd, signalData=metaLearningData)
        metaLearningData = specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData)
        assert specificLayerCounter + 1 == len(specificModel.neuralLayers), f"The specific layer counter ({specificLayerCounter}) does not match the number of specific layers ({len(specificModel.neuralLayers)})."
        # metaLearningData: batchSize, numSignals, finalDimension

        return metaLearningData

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, inferenceTraining):
        # Preallocate the output tensors.
        numExperiments, numSignals, maxSequenceLength, numChannels = signalData.size()
        basicEmotionProfile = torch.zeros((numExperiments, self.numBasicEmotions, self.encodedDimension), device=device, dtype=torch.float64)
        emotionProfile = torch.zeros((numExperiments, self.numEmotions, self.encodedDimension), device=device, dtype=torch.float64)
        reconstructedSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength), device=device, dtype=torch.float64)
        resampledSignalData = torch.zeros((numExperiments, numSignals, self.encodedDimension), device=device, dtype=torch.float64)
        validDataMask = torch.zeros((numExperiments, numSignals, maxSequenceLength), device=device, dtype=torch.bool)
        physiologicalProfile = torch.zeros((numExperiments, self.encodedDimension), device=device, dtype=torch.float64)
        activityProfile = torch.zeros((numExperiments, self.encodedDimension), device=device, dtype=torch.float64)
        testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
        startBatchInd = 0

        while startBatchInd < numExperiments:
            endBatchInd = startBatchInd + testingBatchSize

            # Perform a full pass of the model.
            validDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], resampledSignalData[startBatchInd:endBatchInd], physiologicalProfile[startBatchInd:endBatchInd], \
                activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                = self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd],
                               metadata=metadata[startBatchInd:endBatchInd], device=device, inferenceTraining=inferenceTraining)

            # Update the batch index.
            startBatchInd = endBatchInd

        return validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    def visualizeSignalEncoding(self, physiologicalProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask):
        # Find the first valid signal.
        validSignalMask = torch.any(validDataMask, dim=-1)
        firstBatchInd, firstSignalInd = validSignalMask.nonzero(as_tuple=False)[0, :]
        validPointMask = validDataMask[firstBatchInd, firstSignalInd]

        # Optionally, plot the physiological profile for visual comparison
        physiologicalTimes = self.sharedSignalEncoderModel.pseudoEncodedTimes.detach().cpu().numpy()
        plt.plot(physiologicalTimes, physiologicalProfile[firstBatchInd].detach().cpu().numpy(), 'k', linewidth=1, label='Physiological Profile', alpha=0.75)
        plt.show()

        # Get the first valid signal points.
        validReconstructedPoints = reconstructedSignalData[firstBatchInd, firstSignalInd, validPointMask].detach().cpu().numpy()
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        validTimepoints = timepoints[firstBatchInd, firstSignalInd, validPointMask].detach().cpu().numpy()
        validDatapoints = datapoints[firstBatchInd, firstSignalInd, validPointMask].detach().cpu().numpy()

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(validTimepoints, validDatapoints, 'ok', markersize=3, label='Initial Signal', alpha=0.75)
        plt.plot(validTimepoints, validReconstructedPoints, 'o', color='tab:red', markersize=3, label='Reconstructed Signal', alpha=0.75)
        plt.plot(physiologicalTimes, resampledSignalData[firstBatchInd, firstSignalInd, :].detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Resampled Signal', alpha=0.75)
        plt.title(f"{firstBatchInd} {firstSignalInd} {len(validTimepoints)} {validTimepoints[0]} {validTimepoints[-1]} {len(physiologicalTimes)}")
        plt.legend()
        plt.show()

    def interpolateData(self, signalData, resampledSignalData):
        # Extract the dimensions of the data.
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        batchSize, numSignals, encodedDimension = resampledSignalData.size()

        # Align the timepoints to the physiological times.
        reversedPhysiologicalTimes = torch.flip(self.sharedSignalEncoderModel.pseudoEncodedTimes, dims=[0])
        mappedPhysiologicalTimedInds = encodedDimension - 1 - torch.searchsorted(sorted_sequence=reversedPhysiologicalTimes, input=timepoints, out=None, out_int32=False, right=False)  # timepoints <= physiologicalTimesExpanded[mappedPhysiologicalTimedInds]
        # Ensure the indices don't exceed the size of the last dimension of reconstructedSignalData.
        validIndsRight = torch.clamp(mappedPhysiologicalTimedInds, min=0, max=encodedDimension - 1)  # physiologicalTimesExpanded[validIndsLeft] < timepoints
        validIndsLeft = torch.clamp(mappedPhysiologicalTimedInds + 1, min=0, max=encodedDimension - 1)  # timepoints <= physiologicalTimesExpanded[validIndsRight]
        # mappedPhysiologicalTimedInds dimension: batchSize, numSignals, maxSequenceLength

        # Get the closest physiological data to the timepoints.
        physiologicalTimesExpanded = self.sharedSignalEncoderModel.pseudoEncodedTimes.view(1, 1, -1).expand_as(resampledSignalData)
        closestPhysiologicalTimesRight = torch.gather(input=physiologicalTimesExpanded, dim=2, index=validIndsRight)  # Initialize the tensor.
        closestPhysiologicalTimesLeft = torch.gather(input=physiologicalTimesExpanded, dim=2, index=validIndsLeft)  # Initialize the tensor.
        closestPhysiologicalDataRight = torch.gather(input=resampledSignalData, dim=2, index=validIndsRight)  # Initialize the tensor.
        closestPhysiologicalDataLeft = torch.gather(input=resampledSignalData, dim=2, index=validIndsLeft)  # Initialize the tensor.
        assert ((closestPhysiologicalTimesLeft <= timepoints + 0.1) & (timepoints - 0.1 <= closestPhysiologicalTimesRight)).all(), "The timepoints must be within the range of the closest physiological times."
        # closestPhysiologicalData dimension: batchSize, numSignals, maxSequenceLength

        # Perform linear interpolation.
        linearSlopes = (closestPhysiologicalDataRight - closestPhysiologicalDataLeft) / (closestPhysiologicalTimesRight - closestPhysiologicalTimesLeft).clamp(min=1e-20)
        linearSlopes[closestPhysiologicalTimesLeft == closestPhysiologicalTimesRight] = 0

        # Calculate the error in signal reconstruction (encoding loss).
        interpolatedData = closestPhysiologicalDataLeft + (timepoints - closestPhysiologicalTimesLeft) * linearSlopes

        return interpolatedData

    def reconstructPhysiologicalProfile(self, resampledSignalData):
        reversibleInterface.changeDirections(forwardDirection=True)
        reconstructedPhysiologicalProfile = self.coreModelPass(self.numSignalEncoderLayers, resampledSignalData, specificModel=self.specificSignalEncoderModel, sharedModel=self.sharedSignalEncoderModel)
        reversibleInterface.changeDirections(forwardDirection=False)

        return reconstructedPhysiologicalProfile
