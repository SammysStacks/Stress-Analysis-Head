import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.classWeightHelpers import classWeightHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class emotionDataInterface:
    
    # ---------------------- Labeled Data Getters ---------------------- #

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
        numExperiments, numLabels = allTrainingMasks.size()

        # Find the label indices with training points.
        numTrainingPoints = allTrainingMasks.sum(dim=0)  # Find the number of training points per label.
        goodExperimentalInds = (numTrainingPoints.nonzero(as_tuple=True)[0][numTrainingPoints < numExperiments])
        # If there are no good indices, return an empty list.
        if len(goodExperimentalInds) == 0: return []

        return goodExperimentalInds

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
        signalIdentifiers = inputData[:, :, maxSequencePoints:maxSequencePoints+numSignalIdentifiers, 0].clone()
        metadata = inputData[:, 0, maxSequencePoints+numSignalIdentifiers:, 0].clone()
        signalData = inputData[:, :, 0:maxSequencePoints, :].clone()
        # signalData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # signalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
        # metadata dimension: batchSize, numMetadata

        return signalData, signalIdentifiers, metadata

    @staticmethod
    def separateMaskInformation(trainingMask, numLabels):
        # Separate the training mask information.
        labelMask = trainingMask[:, 0:numLabels]
        signalMask = trainingMask[:, numLabels:]

        return labelMask, signalMask

    # ---------------------- Signal Data Getters ---------------------- #

    @staticmethod
    def getValidDataMask(allSignalData):
        return torch.as_tensor((allSignalData[:, :, :, 0] != 0) & (allSignalData[:, :, :, 1] != 0), device=allSignalData.device)

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
