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
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.datasetName = datasetName  # The name of the dataset the model is training on.

        # General parameters.
        self.encodedDimension = userInputParams.get('encodedDimension', 256)  # The dimension of the encoded signal.
        self.operatorType = userInputParams.get('operatorType', None)  # The type of operator to use for the neural operator.
        self.goldenRatio = userInputParams.get('goldenRatio', 16)  # The number of shared layers per specific layer.
        self.debugging = True

        # Signal encoder parameters.
        self.reversibleLearningProtocol = userInputParams.get('reversibleLearningProtocol', None)   # The learning protocol for the model.
        self.neuralOperatorParameters = userInputParams.get('neuralOperatorParameters', None)   # The parameters for the neural operator.
        self.numLiftingLayersSignalEncoder = 1  # The number of lifting layers to use in the signal encoder.
        self.numSignalEncoderLayers = 0  # The number of layers in the model.

        # Emotion and activity parameters.
        self.irreversibleLearningProtocol = userInputParams.get('irreversibleLearningProtocol', None)  # The learning protocol for the model.
        self.numActivityChannels = userInputParams.get('numActivityChannels', None)  # The number of activity channels to predict.
        self.numBasicEmotions = userInputParams.get('numBasicEmotions', 6)  # The number of basic emotions (basis states of emotions).

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
            encodedDimension=self.encodedDimension,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
            goldenRatio=self.goldenRatio,
            numSignals=self.numSignals,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numLiftingLayers=self.numLiftingLayersSignalEncoder,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            operatorType=self.operatorType,
            goldenRatio=self.goldenRatio,
        )

        # Construct the model weights.
        for _ in range(userInputParams.userInputParams.get('numSignalEncoderLayers', 16)): self.addNewSignalEncoderLayer()
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
        # timepoints: [further away from survey (300) -> closest to survey (0)]
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        # signalIdentifiers: [batchSize, numSignals, numSignalIdentifiers]
        # metadata: [batchSize, numMetadata]

        # Prepare the data for the model.
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.int()
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)

        # ------------------- Organize the Incoming Data ------------------- #

        # Check which points were missing in the data.
        numSignalPoints = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.numSignalPointsSI)  # Dim: batchSize, numSignals
        validDataMask = emotionDataInterface.getValidDataMask(signalData, numSignalPoints)
        # missingDataMask: batchSize, numSignals, maxSequenceLength

        # ------------------- Estimated Physiological Profile ------------------- #

        # Get the estimated physiological profiles.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        if inferenceTraining: physiologicalProfile = self.inferenceModel.getCurrentPhysiologicalProfile(batchInds)
        else: physiologicalProfile = self.specificSignalEncoderModel.getCurrentPhysiologicalProfile(batchInds)
        # physiologicalProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        reversibleInterface.changeDirections(forwardDirection=False)
        resampledSignalData = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, numSignals, 1))
        resampledSignalData = self.coreModelPass(self.numSignalEncoderLayers, resampledSignalData, specificModel=self.specificSignalEncoderModel, sharedModel=self.sharedSignalEncoderModel)
        # resampledSignalData: batchSize, numSignals, encodedDimension

        # Resample the signal data.
        physiologicalTimes = self.sharedSignalEncoderModel.pseudoEncodedTimes
        reconstructedSignalData = self.interpolateData(physiologicalTimes, signalData, resampledSignalData)
        # reconstructedSignalData: batchSize, numSignals, maxSequenceLength

        # Visualize the data transformations within signal encoding.
        if not inferenceTraining and random.random() < 0.02:
            with torch.no_grad(): self.visualizeSignalEncoding(physiologicalProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask)

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            # Get the subject-specific indices.
            subjectInds = emotionDataInterface.getMetaDataChannel(metadata, channelName=modelConstants.subjectIndexMD)  # Dim: batchSize

            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            basicEmotionProfile = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, self.numEmotions*self.numBasicEmotions, 1))
            basicEmotionProfile = self.coreModelPass(metaLearningData=basicEmotionProfile, specificModel=self.specificEmotionModel, sharedModel=self.sharedEmotionModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

            # Reconstruct the emotion data.
            basicEmotionProfile = basicEmotionProfile.view(batchSize, self.numEmotions, self.numBasicEmotions, self.encodedDimension)
            emotionProfile = self.specificEmotionModel.calculateEmotionProfile(basicEmotionProfile, subjectInds)

        # ------------------- Learned Activity Mapping ------------------- #

            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            activityProfile = self.coreModelPass(metaLearningData=physiologicalProfile, specificModel=self.specificActivityModel, sharedModel=self.sharedActivityModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

        # --------------------------------------------------------------- #

        return validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    @staticmethod
    def coreModelPass(numModelLayers, metaLearningData, specificModel, sharedModel):
        specificLayerCounter = 0

        # For each layer in the model.
        for layerInd in range(numModelLayers):
            # Calculate the estimated physiological profile given each signal.
            if layerInd % specificModel.goldenRatio == 0: metaLearningData = specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData); specificLayerCounter += 1  # Reversible signal-specific layers.
            metaLearningData = sharedModel.learningInterface(layerInd=layerInd, signalData=metaLearningData)  # Reversible meta-learning layers.
        metaLearningData = specificModel.learningInterface(layerInd=specificLayerCounter, signalData=metaLearningData)  # Reversible signal-specific layers.
        assert specificLayerCounter + 1 == len(specificModel.neuralLayers), f"The specific layer counter ({specificLayerCounter}) does not match the number of specific layers ({len(specificModel.neuralLayers)})."
        # metaLearningData: batchSize, numSignals, finalDimension

        return metaLearningData

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, inferenceTraining):
        with torch.no_grad():
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
        plt.plot(physiologicalTimes, physiologicalProfile[0].detach().cpu().numpy(), 'k', linewidth=1, label='Physiological Profile', alpha=0.75)
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

    @staticmethod
    def interpolateData(physiologicalTimes, signalData, resampledSignalData):
        # Extract the dimensions of the data.
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        batchSize, numSignals, encodedDimension = resampledSignalData.size()

        # Align the timepoints to the physiological times.
        reversedPhysiologicalTimes = torch.flip(physiologicalTimes, dims=[0])
        mappedPhysiologicalTimedInds = encodedDimension - 1 - torch.searchsorted(sorted_sequence=reversedPhysiologicalTimes, input=timepoints, out=None, out_int32=False, right=False)  # timepoints <= physiologicalTimesExpanded[mappedPhysiologicalTimedInds]
        # Ensure the indices don't exceed the size of the last dimension of reconstructedSignalData.
        validIndsRight = torch.clamp(mappedPhysiologicalTimedInds, min=0, max=encodedDimension - 1)  # physiologicalTimesExpanded[validIndsLeft] < timepoints
        validIndsLeft = torch.clamp(mappedPhysiologicalTimedInds + 1, min=0, max=encodedDimension - 1)  # timepoints <= physiologicalTimesExpanded[validIndsRight]
        # mappedPhysiologicalTimedInds dimension: batchSize, numSignals, maxSequenceLength

        # Get the closest physiological data to the timepoints.
        physiologicalTimesExpanded = physiologicalTimes.unsqueeze(0).unsqueeze(0).expand(batchSize, numSignals, encodedDimension)
        closestPhysiologicalTimesRight = torch.gather(input=physiologicalTimesExpanded, dim=2, index=validIndsRight)  # Initialize the tensor.
        closestPhysiologicalTimesLeft = torch.gather(input=physiologicalTimesExpanded, dim=2, index=validIndsLeft)  # Initialize the tensor.
        closestPhysiologicalDataRight = torch.gather(input=resampledSignalData, dim=2, index=validIndsRight)  # Initialize the tensor.
        closestPhysiologicalDataLeft = torch.gather(input=resampledSignalData, dim=2, index=validIndsLeft)  # Initialize the tensor.
        assert ((closestPhysiologicalTimesLeft <= timepoints) & (timepoints <= closestPhysiologicalTimesRight)).all(), "The timepoints must be within the range of the closest physiological times."
        # closestPhysiologicalData dimension: batchSize, numSignals, maxSequenceLength

        # Perform linear interpolation.
        linearSlopes = (closestPhysiologicalDataRight - closestPhysiologicalDataLeft) / (closestPhysiologicalTimesRight - closestPhysiologicalTimesLeft).clamp(min=1e-8)
        linearSlopes[closestPhysiologicalTimesLeft == closestPhysiologicalTimesRight] = 0

        # Calculate the error in signal reconstruction (encoding loss).
        interpolatedData = closestPhysiologicalDataLeft + (timepoints - closestPhysiologicalTimesLeft) * linearSlopes

        return interpolatedData

    def reconstructPhysiologicalProfile(self, resampledSignalData):
        reversibleInterface.changeDirections(forwardDirection=True)
        reconstructedPhysiologicalProfile = self.coreModelPass(self.numSignalEncoderLayers, resampledSignalData, specificModel=self.specificSignalEncoderModel, sharedModel=self.sharedSignalEncoderModel)
        reversibleInterface.changeDirections(forwardDirection=False)

        return reconstructedPhysiologicalProfile
