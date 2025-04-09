import torch

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

    @staticmethod
    def getActivityLabels(allLabels, allLabelsMask, activityLabelInd):
        activityMask = emotionDataInterface.getActivityColumn(allLabelsMask, activityLabelInd)
        allActivityLabels = emotionDataInterface.getActivityColumn(allLabels, activityLabelInd)

        return allActivityLabels[activityMask]

    @staticmethod
    def getEmotionLabels(emotionInd, allLabels, allLabelsMask):
        emotionDataMask = emotionDataInterface.getEmotionColumn(allLabelsMask, emotionInd)
        emotionLabels = emotionDataInterface.getEmotionColumn(allLabels, emotionInd)

        return emotionLabels[emotionDataMask]

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

    # ---------------------- Emotion and Activity Model ---------------------- #

    @staticmethod
    def getFullGaussianProfile(encodedDimension, device, numClasses):
        gaussianWeight = torch.zeros(encodedDimension, device=device)

        # For each class.
        for classInd in range(numClasses):
            # Add the Gaussian weights for the predicted class profile.
            gaussianWeight += emotionDataInterface.getGaussianWeights(encodedDimension, device, numClasses, classInd)[1]
        gaussianWeight = gaussianWeight / gaussianWeight.sum()  # Normalize the distribution.

        return gaussianWeight

    @staticmethod
    def getGaussianWeights(encodedDimension, device, numClasses, classInd):
        classDimension = encodedDimension // numClasses

        # Generate the Gaussian weights for the predicted activity profile.
        gaussianWeight = emotionDataInterface.gaussian_1d_kernel(encodedDimension, classInd*classDimension + classDimension / 2, classDimension / 6, device=device)
        return classDimension, gaussianWeight

    @staticmethod
    def gaussian_1d_kernel(size, mu, std, device):
        # Create an array of indices and shift them so that the mean is at mu.
        x = torch.arange(0, size, dtype=torch.float32, device=device) - mu
        # Calculate the Gaussian function for each index.
        kernel = torch.exp(-(x ** 2) / (2 * std ** 2))
        # Normalize the kernel so that the sum of all values is 1.
        kernel = kernel / kernel.sum()

        return kernel

    @staticmethod
    def getActivityClassProfile(predictedActivityProfile, numActivities):
        batchSize, encodedDimension = predictedActivityProfile.shape
        device = predictedActivityProfile.device

        # Get the full Gaussian profile for the activity classes.
        gaussianWeightProfile = emotionDataInterface.getFullGaussianProfile(encodedDimension, device=device, numClasses=numActivities)
        classDimension = encodedDimension // numActivities

        # Set up the predicted activity classes.
        weightActivityProfile = predictedActivityProfile - predictedActivityProfile.min(dim=-1, keepdim=True).values
        weightActivityProfile = weightActivityProfile * gaussianWeightProfile  # TODO: normalization needed?
        predictedActivityClasses = torch.zeros(batchSize, numActivities, device=device)

        # For each activity class.
        for classInd in range(numActivities):
            predictedActivityClasses[:, classInd] = weightActivityProfile[:, classInd*classDimension:(classInd+1)*classDimension].sum(dim=-1)

        return predictedActivityClasses

    @staticmethod
    def getEmotionClassPredictions(predictedEmotionProfile, allEmotionClasses, device):
        batchSize, numEmotions, encodedDimension = predictedEmotionProfile.shape
        allEmotionClassPredictions = []

        for emotionInd in range(numEmotions):
            # Get the relevant batches for the current emotion.
            emotionProfile = predictedEmotionProfile[:, emotionInd]  # Dim: batchSize, encodedDimension
            numEmotionClasses = allEmotionClasses[emotionInd]

            # Set up the predicted emotion classes.
            emotionClassPredictions = torch.zeros(batchSize, numEmotionClasses, device=device, dtype=predictedEmotionProfile.dtype)
            classDimension = encodedDimension // numEmotionClasses

            # Get the full Gaussian weight profile for the emotion classes.
            gaussianWeightProfile = emotionDataInterface.getFullGaussianProfile(encodedDimension, device=device, numClasses=numEmotionClasses)
            weightedProfile = emotionProfile * gaussianWeightProfile

            # For each emotion class.
            for classInd in range(numEmotionClasses):
                emotionClassPredictions[:, classInd] = weightedProfile[:, classInd*classDimension:(classInd + 1)*classDimension].sum(dim=-1)
            emotionClassPredictions = emotionClassPredictions / emotionClassPredictions.sum(dim=-1, keepdim=True)  # Normalize the distribution.

            # Store the predicted emotion classes.
            allEmotionClassPredictions.append(emotionClassPredictions)

        return allEmotionClassPredictions

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
