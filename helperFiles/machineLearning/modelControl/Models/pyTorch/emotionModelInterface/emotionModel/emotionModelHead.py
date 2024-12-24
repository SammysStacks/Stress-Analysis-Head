import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.lossInformation.lossCalculations import lossCalculations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedActivityModel import sharedActivityModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificActivityModel import specificActivityModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel


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
        self.debugging = True

        # Signal encoder parameters.
        self.reversibleLearningProtocol = userInputParams['reversibleLearningProtocol']   # The learning protocol for the model.
        self.neuralOperatorParameters = userInputParams['neuralOperatorParameters']   # The parameters for the neural operator.
        self.numSpecificEncoderLayers = userInputParams['numSpecificEncoderLayers']  # The number of specific layers.
        self.numSharedEncoderLayers = userInputParams['numSharedEncoderLayers']  # The number of shared layers.

        # Emotion and activity parameters.
        self.irreversibleLearningProtocol = userInputParams['irreversibleLearningProtocol']  # The learning protocol for the model.
        self.numActivityModelLayers = userInputParams['numActivityModelLayers']  # The number of basic emotions (basis states of emotions).
        self.numEmotionModelLayers = userInputParams['numEmotionModelLayers']  # The number of basic emotions (basis states of emotions).
        self.numActivityChannels = userInputParams['numActivityChannels']  # The number of activity channels to predict.
        self.numBasicEmotions = userInputParams['numBasicEmotions']  # The number of basic emotions (basis states of emotions).

        # Setup holder for the model's training information
        self.calculateModelLosses = lossCalculations(accelerator=None, allEmotionClasses=None, activityLabelInd=None)
        self.reversibleInterface = reversibleInterface()

        # ------------------------ Data Compression ------------------------ #

        # The signal encoder model to find a common feature vector across all signals.
        self.specificSignalEncoderModel = specificSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numSpecificEncoderLayers=self.numSpecificEncoderLayers,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            featureNames=self.featureNames,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numSharedEncoderLayers=self.numSharedEncoderLayers,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            operatorType=self.operatorType,
        )

        # -------------------- Final Emotion Prediction -------------------- #

        if submodel == modelConstants.emotionModel:
            self.specificEmotionModel = specificEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numEmotionModelLayers,
                operatorType=self.operatorType,
                numSpecificEncoderLayers=self.numSpecificEncoderLayers,
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
                numSpecificEncoderLayers=self.numSpecificEncoderLayers,
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

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, onlyProfileTraining=False):
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

        # Get the estimated physiological weights.
        embeddedProfile = self.specificSignalEncoderModel.profileModel.getPhysiologicalProfile(batchInds)
        physiologicalProfile = self.sharedSignalEncoderModel.smoothPhysiologicalProfile(embeddedProfile)
        # embeddedProfile: batchSize, modelConstants.numEncodedWeights
        # physiologicalProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Perform the backward pass: physiological profile -> signal data.
        resampledSignalData = physiologicalProfile.unsqueeze(1).repeat(repeats=(1, numSignals, 1))
        resampledSignalData, compiledSignalEncoderLayerState = self.signalEncoderPass(metaLearningData=resampledSignalData, forwardPass=False, compileLayerStates=onlyProfileTraining)
        # resampledSignalData: batchSize, numSignals, encodedDimension

        # Resample the signal data.
        reconstructedSignalData = self.sharedSignalEncoderModel.interpolateData(signalData, resampledSignalData)
        # reconstructedSignalData: batchSize, numSignals, maxSequenceLength

        # Visualize the data transformations within signal encoding.
        if submodel == modelConstants.signalEncoderModel and not onlyProfileTraining and random.random() < 0.01:
            with torch.no_grad(): self.visualizeSignalEncoding(embeddedProfile, physiologicalProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask)

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

        return validDataMask, reconstructedSignalData, resampledSignalData, compiledSignalEncoderLayerState, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    # ------------------------- Model Components ------------------------- #

    def signalEncoderPass(self, metaLearningData, forwardPass, compileLayerStates=False):
        # Initialize the model's learning interface.
        compiledLayerStates = [metaLearningData.clone().detach().cpu().numpy()] if compileLayerStates else None
        reversibleInterface.changeDirections(forwardDirection=forwardPass)

        # Calculate the estimated physiological profile given each signal.
        for specificLayerInd in range(self.numSpecificEncoderLayers):
            metaLearningData = self.specificSignalEncoderModel.learningInterface(layerInd=specificLayerInd, signalData=metaLearningData)
            if compileLayerStates: compiledLayerStates.append(metaLearningData.clone().detach().cpu().numpy())
        # metaLearningData: batchSize, numSignals, finalDimension

        # Calculate the estimated physiological profile given each signal.
        for sharedLayerInd in range(self.numSharedEncoderLayers): 
            metaLearningData = self.sharedSignalEncoderModel.learningInterface(layerInd=sharedLayerInd, signalData=metaLearningData)
            if compileLayerStates: compiledLayerStates.append(metaLearningData.clone().detach().cpu().numpy())
        # metaLearningData: batchSize, numSignals, finalDimension

        # Calculate the estimated physiological profile given each signal.
        for specificLayerInd in range(self.numSpecificEncoderLayers, 2*self.numSpecificEncoderLayers): 
            metaLearningData = self.specificSignalEncoderModel.learningInterface(layerInd=specificLayerInd, signalData=metaLearningData)
            if compileLayerStates: compiledLayerStates.append(metaLearningData.clone().detach().cpu().numpy())
        # metaLearningData: batchSize, numSignals, finalDimension

        # Compile the layer states if necessary.
        if compileLayerStates: compiledLayerStates = np.asarray(compiledLayerStates)
        return metaLearningData, compiledLayerStates

    def reconstructPhysiologicalProfile(self, resampledSignalData):
        return self.signalEncoderPass(metaLearningData=resampledSignalData, forwardPass=True, compileLayerStates=True)

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, profileEpoch=None):
        # Preallocate the output tensors.
        numExperiments, numSignals, maxSequenceLength, numChannels = signalData.size()
        testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
        onlyProfileTraining = profileEpoch is not None

        with torch.no_grad():
            # Initialize the output tensors.
            compiledSignalEncoderLayerStates = np.zeros(shape=(2*self.numSpecificEncoderLayers + self.numSharedEncoderLayers + 1, numExperiments, 1, self.encodedDimension))
            basicEmotionProfile = torch.zeros((numExperiments, self.numBasicEmotions, self.encodedDimension), device=device)
            validDataMask = torch.zeros((numExperiments, numSignals, maxSequenceLength), device=device, dtype=torch.bool)
            emotionProfile = torch.zeros((numExperiments, self.numEmotions, self.encodedDimension), device=device)
            reconstructedSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength), device=device)
            resampledSignalData = torch.zeros((numExperiments, numSignals, self.encodedDimension), device=device)
            physiologicalProfile = torch.zeros((numExperiments, self.encodedDimension), device=device)
            activityProfile = torch.zeros((numExperiments, self.encodedDimension), device=device)

        startBatchInd = 0
        while startBatchInd < numExperiments:
            endBatchInd = startBatchInd + testingBatchSize

            # Perform a full pass of the model.
            validDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], resampledSignalData[startBatchInd:endBatchInd], compiledSignalEncoderLayerState, \
                physiologicalProfile[startBatchInd:endBatchInd], activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                = self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd],
                               metadata=metadata[startBatchInd:endBatchInd], device=device, onlyProfileTraining=onlyProfileTraining)
            if compiledSignalEncoderLayerState is not None: assert onlyProfileTraining; compiledSignalEncoderLayerStates[:, startBatchInd:endBatchInd, -1, :] = compiledSignalEncoderLayerState[:, :, -1, :]

            # Update the batch index.
            startBatchInd = endBatchInd

        if onlyProfileTraining:
            with torch.no_grad():
                batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0].long()  # Dim: batchSize
                batchLossValues = self.calculateModelLosses.calculateSignalEncodingLoss(signalData, reconstructedSignalData, validDataMask, allSignalMask=None, averageBatches=False)
                self.specificSignalEncoderModel.profileModel.populateProfileState(profileEpoch, batchInds, batchLossValues, physiologicalProfile, compiledSignalEncoderLayerStates)

        return validDataMask, reconstructedSignalData, resampledSignalData, compiledSignalEncoderLayerStates, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile

    # ------------------------- Model Visualizations ------------------------- #

    def visualizeSignalEncoding(self, embeddedProfile, physiologicalProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask):
        # Find the first valid signal.
        validSignalMask = torch.any(validDataMask, dim=-1)
        firstBatchInd, firstSignalInd = validSignalMask.nonzero(as_tuple=False)[0, :]
        validPointMask = validDataMask[firstBatchInd, firstSignalInd]

        # Optionally, plot the physiological profile for visual comparison
        resampledBiomarkerTimes = self.sharedSignalEncoderModel.hyperSampledTimes.clone().detach().cpu().numpy()
        plt.plot(resampledBiomarkerTimes, physiologicalProfile[firstBatchInd].clone().detach().cpu().numpy(), 'tab:red', linewidth=1, label='Physiological Profile', alpha=2/3)
        plt.plot(torch.linspace(start=resampledBiomarkerTimes[0], end=resampledBiomarkerTimes[-1], steps=embeddedProfile.size(-1)).clone().detach().cpu().numpy(), embeddedProfile[firstBatchInd].clone().detach().cpu().numpy(), 'ok', linewidth=1, markersize=3,  label='Original Profile', alpha=0.75)
        plt.title(f"batchInd{firstBatchInd}")
        plt.ylim((-1/3, 1/3))
        plt.show()

        # Get the first valid signal points.
        validReconstructedPoints = reconstructedSignalData[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        validTimepoints = timepoints[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()
        validDatapoints = datapoints[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(validTimepoints, validDatapoints, 'ok', markersize=3, label='Initial Signal', alpha=0.75)
        plt.plot(validTimepoints, validReconstructedPoints, 'o', color='tab:red', markersize=3, label='Reconstructed Signal', alpha=0.75)
        plt.plot(resampledBiomarkerTimes, resampledSignalData[firstBatchInd, firstSignalInd, :].clone().detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Resampled Signal', alpha=0.75)
        plt.title(f"batchInd{firstBatchInd} signalInd{firstSignalInd} numPoints{len(validTimepoints)}")
        plt.ylim((-1.5, 1.5))
        plt.legend()
        plt.show()
