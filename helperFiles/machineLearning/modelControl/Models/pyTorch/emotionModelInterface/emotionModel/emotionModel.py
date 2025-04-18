import random

import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHead import emotionModelHead
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class emotionModel(emotionModelHead):
    def __init__(self, submodel, emotionNames, activityNames, featureNames, allEmotionClasses, numSubjects, datasetName, numExperiments):
        super(emotionModel, self).__init__(submodel, emotionNames, activityNames, featureNames, allEmotionClasses, numSubjects, datasetName, numExperiments)

    # ------------------------- Full Forward Calls ------------------------- #

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, compiledLayerStates=False):
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.int()
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        # timepoints: [further away from survey (300) -> closest to survey (0)]
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        # signalIdentifiers: [batchSize, numSignals, numSignalIdentifiers]
        # metadata: [batchSize, numMetadata]

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numEmotions, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        reconstructedSignalData = torch.zeros((batchSize, numSignals, maxSequenceLength), device=device)
        resampledSignalData = torch.zeros((batchSize, numSignals, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)

        # ------------------- Organize the Incoming Data ------------------- #

        # Check which points were missing in the data.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        validDataMask = emotionDataInterface.getValidDataMask(signalData)
        # validDataMask: batchSize, numSignals, maxSequenceLength

        # ------------------- Estimated Health Profile ------------------- #

        # Get the estimated profile weights.
        embeddedProfile = self.specificSignalEncoderModel.healthEmbeddingModel(batchInds, self.sharedSignalEncoderModel.fourierModel)
        healthProfile = self.sharedSignalEncoderModel.generateHealthProfile(embeddedProfile)
        # embeddedProfile: batchSize, encodedDimension
        # healthProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        if submodel == modelConstants.signalEncoderModel:
            # Perform the backward pass: health profile -> signal data.
            resampledSignalData = healthProfile.unsqueeze(1).repeat(repeats=(1, numSignals, 1))
            resampledSignalData = self.signalEmbedding(metaLearningData=resampledSignalData, healthProfileStart=True, compileLayerStates=compiledLayerStates)
            reconstructedSignalData = self.sharedSignalEncoderModel.interpolateOriginalSignals(signalData, resampledSignalData)
            # reconstructedSignalData: batchSize, numSignals, maxSequenceLength
            # resampledSignalData: batchSize, numSignals, encodedDimension

            if random.random() < 0.01 and not self.hpcFlag:
                # Visualize the data transformations within signal encoding.
                with torch.no_grad(): self.visualizeSignalEncoding(embeddedProfile, healthProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask)

        # ------------- Learned Emotion and Activity Mapping ------------- #

        if submodel == modelConstants.emotionModel:
            # Perform the backward pass: health profile -> activity data.
            activityProfile = self.humanActivityRecognition(healthProfile=healthProfile.unsqueeze(1)).squeeze(1)
            # activityProfile: batchSize, encodedDimension

            # Perform the backward pass: health profile -> basic emotion data.
            basicEmotionProfile = self.basicEmotionProfiling(healthProfile=healthProfile)
            # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension

            # Reconstruct the emotion data.
            subjectInds = emotionDataInterface.getMetaDataChannel(metadata, channelName=modelConstants.subjectIndexMD)  # Dim: batchSize
            emotionProfile = self.specificEmotionModel.calculateEmotionProfile(basicEmotionProfile, subjectInds)
            # emotionProfile: batchSize, numEmotions, encodedDimension

        # --------------------------------------------------------------- #

        return validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, profileEpoch=None):
        # Preallocate the output tensors.
        numExperiments, numSignals, maxSequenceLength, numChannels = signalData.size()
        testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
        onlyProfileTraining = profileEpoch is not None

        # Initialize the output tensors.
        basicEmotionProfile = torch.zeros((numExperiments, self.numEmotions, self.numBasicEmotions, self.encodedDimension), device='cpu')
        validDataMask = torch.zeros((numExperiments, numSignals, maxSequenceLength), device='cpu', dtype=torch.bool)
        emotionProfile = torch.zeros((numExperiments, self.numEmotions, self.encodedDimension), device='cpu')
        reconstructedSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength), device='cpu')
        resampledSignalData = torch.zeros((numExperiments, numSignals, self.encodedDimension), device='cpu')
        activityProfile = torch.zeros((numExperiments, self.encodedDimension), device='cpu')
        healthProfile = torch.zeros((numExperiments, self.encodedDimension), device='cpu')

        startBatchInd = 0
        while startBatchInd < numExperiments:
            endBatchInd = startBatchInd + testingBatchSize

            # Perform a full pass of the model.
            validDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], resampledSignalData[startBatchInd:endBatchInd], \
                healthProfile[startBatchInd:endBatchInd], activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                = (element.cpu() if isinstance(element, torch.Tensor) else element for element in self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd],
                                                                                                               metadata=metadata[startBatchInd:endBatchInd], device=device, compiledLayerStates=False))
            startBatchInd = endBatchInd  # Update the batch index.

        if onlyProfileTraining:
            with torch.no_grad():
                batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0].long()  # Dim: batchSize
                batchLossValues = self.calculateModelLosses.calculateSignalEncodingLoss(signalData.cpu(), reconstructedSignalData, validDataMask, allSignalMask=None, averageBatches=False)
                self.specificSignalEncoderModel.profileModel.populateProfileState(profileEpoch, batchInds, batchLossValues, resampledSignalData, healthProfile)

        return validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile

    # ------------------------- Model Components ------------------------- #

    def signalEmbedding(self, metaLearningData, healthProfileStart, compileLayerStates=False):
        if compileLayerStates: self.specificSignalEncoderModel.profileModel.resetModelStates(metaLearningData)  # Add the initial state.
        compilingFunction = self.specificSignalEncoderModel.profileModel.addModelState if compileLayerStates else None
        reversibleInterface.changeDirections(forwardDirection=not healthProfileStart)

        if not healthProfileStart:
            metaLearningData = self.sharedSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            metaLearningData = self.specificSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            # metaLearningData: batchSize, numSignals, encodedDimension
        else:
            metaLearningData = self.specificSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            metaLearningData = self.sharedSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            # metaLearningData: batchSize, numSignals, encodedDimension

        return metaLearningData

    def humanActivityRecognition(self, healthProfile):
        reversibleInterface.changeDirections(forwardDirection=False)

        # Learn the activity profile.
        metaLearningData = self.specificActivityModel.learningInterface(signalData=healthProfile)
        metaLearningData = self.sharedActivityModel.learningInterface(signalData=metaLearningData)
        # metaLearningData: batchSize, 1, encodedDimension

        return metaLearningData

    def basicEmotionProfiling(self, healthProfile):
        reversibleInterface.changeDirections(forwardDirection=False)
        batchSize, encodedDimension = healthProfile.size()

        # Create a basic emotion profile.
        basicEmotionProfile = healthProfile.unsqueeze(1).unsqueeze(1)
        basicEmotionProfile = basicEmotionProfile.repeat(repeats=(1, self.numEmotions, self.numBasicEmotions, 1))
        # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension

        # Apply a specific emotion layer to the data.
        basicEmotionProfile = basicEmotionProfile.view(batchSize, self.numEmotions*self.numBasicEmotions, self.encodedDimension)
        basicEmotionProfile = self.specificEmotionModel.learningInterface(signalData=basicEmotionProfile)
        basicEmotionProfile = basicEmotionProfile.view(batchSize, self.numEmotions, self.numBasicEmotions, self.encodedDimension)
        # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension

        # Learn the basic emotion profile.
        basicEmotionProfile = basicEmotionProfile.view(batchSize*self.numEmotions, self.numBasicEmotions, self.encodedDimension)
        basicEmotionProfile = self.sharedEmotionModel.learningInterface(signalData=basicEmotionProfile)
        basicEmotionProfile = basicEmotionProfile.view(batchSize, self.numEmotions, self.numBasicEmotions, self.encodedDimension)
        # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension

        return basicEmotionProfile

    def reconstructHealthProfile(self, resampledSignalData):
        return self.signalEmbedding(metaLearningData=resampledSignalData, healthProfileStart=False, compileLayerStates=True)

    # ------------------------- Model Updates ------------------------- #

    def cullAngles(self, epoch):
        for name, module in self.named_modules():
            sharedLayer = 'shared' in name.lower()
            if isinstance(module, reversibleLieLayer):
                module.angularThresholding(epoch=epoch, sharedLayer=sharedLayer)
