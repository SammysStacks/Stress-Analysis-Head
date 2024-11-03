# Helper classes
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.classWeightHelpers import classWeightHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class emotionDataInterface:
    
    # ---------------------- Labeled Data Getters ---------------------- #

    @staticmethod
    def getReconstructionIndex(allTrainingMasks):
        # Find the first label index with training points.
        reconstructionIndices = emotionDataInterface.getLabelInds_withPoints(allTrainingMasks)
        assert len(reconstructionIndices) != 0, f"We should have some training data: {reconstructionIndices}"

        return reconstructionIndices[0]

    @staticmethod
    def getActivityColumn(allLabels, activityLabelInd):
        return allLabels[:, activityLabelInd]

    @staticmethod
    def getEmotionMasks(allLabelsMask, numEmotions):
        return allLabelsMask[:, 0:numEmotions]

    @staticmethod
    def getEmotionColumn(allLabels, emotionInd):
        return allLabels[:, emotionInd]

    def getActivityLabels(self, allLabels, allLabelsMask, activityLabelInd):
        activityMask = self.getActivityColumn(allLabelsMask, activityLabelInd)
        allActivityLabels = self.getActivityColumn(allLabels, activityLabelInd)

        return allActivityLabels[activityMask]

    def getEmotionLabels(self, emotionInd, allLabels, allLabelsMask):
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)
        emotionLabels = self.getEmotionColumn(allLabels, emotionInd)

        return emotionLabels[emotionDataMask]

    def getEmotionDistributions(self, emotionInd, predictedEmotionLabels, allLabels, allLabelsMask, allEmotionClasses, emotionLength):
        # Organize the emotion's training information.
        emotionLabels = self.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)

        # Get the predicted and true emotion distributions.
        predictedTrainingEmotions = predictedEmotionLabels[emotionInd][emotionDataMask]
        trueTrainingEmotions = classWeightHelpers.gausEncoding(emotionLabels, allEmotionClasses[emotionInd], emotionLength)

        return predictedTrainingEmotions, trueTrainingEmotions

    @staticmethod
    def getLabelInds_withPoints(allTrainingMasks):
        # Find the label indices with training points.
        numTrainingPoints = allTrainingMasks.sum(dim=0)  # Find the number of training points per label.
        goodExperimentalInds = numTrainingPoints.nonzero()
        # If there are no good indices, return an empty list.
        if len(goodExperimentalInds) == 0: return []

        return goodExperimentalInds[:, 0]

    # ---------------------- Contextual Data Getters ---------------------- #
    
    @staticmethod
    def getMetadataIndex(metadataName):
        return modelConstants.metadata.index(metadataName)
    
    @staticmethod
    def getMetadata(metadata, metadataName):
        # metadata dim: (batchSize, numMetadata)
        metadataInd = emotionDataInterface.getMetadataIndex(metadataName)

        return metadata[:, metadataInd]
    
    @staticmethod
    def getSignalIdentifierIndex(identifierName):
        return modelConstants.signalIdentifiers.index(identifierName)
    
    @staticmethod
    def getSignalIdentifiers(signalIdentifiers, identifierName):
        # signalIdentifiers dim: (batchSize, numSignals, numSignalIdentifiers)
        signalIdentifierInd = emotionDataInterface.getSignalIdentifierIndex(identifierName)

        return signalIdentifiers[:, :, signalIdentifierInd]

    # ---------------------- Signal Info Getters ---------------------- #

    @staticmethod
    def separateData(inputData):
        # allSignalData dim: (batchSize, numSignals, fullDataLength, [timeChannel, signalChannel])
        # Extract the incoming data's dimension and ensure a proper data format.
        batchSize, numSignals, totalLength, numChannels = inputData.size()

        # Find the amount of contextual information.
        numSignalIdentifiers = len(modelConstants.signalIdentifiers)
        numMetadata = len(modelConstants.metadata)

        # Find the maximum sequence points.
        maxSequencePoints = totalLength - numMetadata - numSignalIdentifiers

        # Separate the sequence and demographic information.
        signalIdentifiers = inputData[:, :, maxSequencePoints:maxSequencePoints+numSignalIdentifiers, 0]
        metadata = inputData[:, 0, maxSequencePoints+numSignalIdentifiers:, 0]
        signalData = inputData[:, :, 0:maxSequencePoints, :]
        # signalData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # signalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
        # metadata dimension: batchSize, numMetadata

        return signalData, signalIdentifiers, metadata

    # ---------------------- Signal Data Getters ---------------------- #

    @staticmethod
    def getValidDataMask(allSignalData, allNumSignalPoints):
        # Extract the incoming data's dimension.
        # batchSize, numSignals, maxSequenceLength = allSignalData.size()[0:3]

        # This creates a range tensor for the sequence dimension.
        # positionTensor = torch.arange(start=0, end=maxSequenceLength, step=1, device=allSignalData.device).expand(batchSize, numSignals, maxSequenceLength)

        # Compare `positionTensor` with `allNumSignalPoints` (broadcast).
        # validDataMask = positionTensor < allNumSignalPoints.unsqueeze(-1)

        # Assert the validity of the data mask.
        potentialDataMask = torch.as_tensor((allSignalData[:, :, :, 0] != 0) & (allSignalData[:, :, :, 1] != 0), device=allSignalData.device)
        # assert (validDataMask == potentialDataMask).all(), "The data mask is not correct."

        return potentialDataMask

    @staticmethod
    def getChannelInd(channelName):
        return modelConstants.signalChannelNames.index(channelName)
    
    @staticmethod
    def getChannelData_fromInd(signalData, channelInd):
        # allSignalData dim: (batchSize, numSignals, fullDataLength, [timeChannel, signalChannel])
        return signalData[:, :, :, channelInd]

    @staticmethod
    def getChannelData(signalData, channelName):
        # allSignalData dim: (batchSize, numSignals, fullDataLength, [timeChannel, signalChannel])
        channelInd = emotionDataInterface.getChannelInd(channelName)
        channelData = emotionDataInterface.getChannelData_fromInd(signalData, channelInd)
        
        return channelData

    @staticmethod
    def getSignalIdentifierData(signalIdentifiers, channelName):
        # signalIdentifiers dim: (batchSize, numSignals, numSignalIdentifiers)
        channelInd = modelConstants.signalIdentifiers.index(channelName)

        return signalIdentifiers[:, :, channelInd]

    @staticmethod
    def getMetaDataChannel(metaData, channelName):
        channelInd = modelConstants.metadata.index(channelName)

        return metaData[:, channelInd]
    
    def getReconstructionData(self, trainingMask, signalLabels, signalData, reconstructionIndex):
        # Get the current training data mask.
        trainingColumnMask = self.getEmotionColumn(trainingMask, reconstructionIndex)

        # Apply the training data mask
        trainingMaskRecon = trainingMask[trainingColumnMask]
        signalLabelsRecon = signalLabels[trainingColumnMask]
        signalDataRecon = signalData[trainingColumnMask]

        return trainingMaskRecon, signalLabelsRecon, signalDataRecon
