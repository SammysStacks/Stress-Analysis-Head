import torch
from matplotlib import pyplot as plt
from torch import nn

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from ..emotionDataInterface import emotionDataInterface
from ..modelConstants import modelConstants


class lossCalculations:

    def __init__(self, accelerator, allEmotionClasses, numActivities, activityLabelInd):
        # General parameters
        self.numEmotions = len(allEmotionClasses or [])  # The number of emotions to predict.
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: numEmotions
        self.activityLabelInd = activityLabelInd  # The index of the activity label in the label tensor.
        self.numActivities = numActivities  # The number of activities to predict. Type: int
        self.accelerator = accelerator  # Hugging face model optimizations.

        # Calculate the number of sequence points to throw out.
        self.lossScaleFactor = 10  # The factor to scale the loss by when removing points.
        self.numCulledLosses = 4  # The number of times to remove the top noisy points.

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()
        self.modelHelpers = modelHelpers()

        # Initialize the loss function WITHOUT the class weights.
        self.smoothL1Loss = nn.SmoothL1Loss(reduction='none', beta=1)  # NEVER MAKE IT LESS THAN 1, its just MAE for outliers.
        self.activityCrossEntropyLoss, self.emotionCrossEntropyLoss = None, None  # The cross-entropy loss functions for the activity and emotion labels.

    def setEmotionActivityLossFunctions(self, activityClassWeights, emotionClassWeights):
        self.activityCrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=activityClassWeights, reduction='none', label_smoothing=0)
        self.emotionCrossEntropyLoss = torch.nn.ModuleList([
            torch.nn.CrossEntropyLoss(weight=emotionClassWeight, reduction='none', label_smoothing=0.1)
            for emotionClassWeight in emotionClassWeights
        ])

    # -------------------------- Signal Encoder Loss Calculations ------------------------- #

    def calculateSignalEncodingLoss(self, allInitialSignalData, allReconstructedSignalData, allValidDataMask, allSignalMask, averageBatches=True):
        # Get the relevant data for the loss calculation.
        allDatapoints = emotionDataInterface.getChannelData(allInitialSignalData, channelName=modelConstants.signalChannel)
        allDatapoints = allDatapoints.to(allReconstructedSignalData.device)
        validDataMask = allValidDataMask.clone()  # Masks out missing data points: batchSize, numSignals, sequenceLength

        # Compile the relevant data for the loss calculation.
        if allSignalMask is not None: validDataMask[~allSignalMask.unsqueeze(-1).expand_as(validDataMask)] = False  # Additionally, masks out training vs. testing.
        if validDataMask.sum() == 0: print("No valid batches"); return None

        # Calculate the error in signal reconstruction (encoding loss).
        signalReconstructedLoss = self.smoothL1Loss(allReconstructedSignalData, allDatapoints)
        self.modelHelpers.assertVariableIntegrity(signalReconstructedLoss, variableName="encoded signal reconstructed loss", assertGradient=False)
        # signalReconstructedLoss: batchSize, numSignals, sequenceLength

        # Create index grids for batch and numSignals dimensions
        batchSize, numSignals, sequenceLength = allDatapoints.size()
        batch_indices, signal_indices = torch.meshgrid(torch.arange(batchSize, device=validDataMask.device), torch.arange(numSignals, device=validDataMask.device), indexing="ij")

        # For each point to remove.
        for _ in range(self.numCulledLosses):
            # Find the index of the maximum loss for each signal.
            findMaxLoss = torch.where(validDataMask, signalReconstructedLoss, float('-inf'))
            max_indices = findMaxLoss.argmax(dim=-1, keepdim=True).squeeze(-1)

            # Reduce the loss for the top noisy points.
            signalReconstructedLoss[batch_indices, signal_indices, max_indices] = signalReconstructedLoss[batch_indices, signal_indices, max_indices] / self.lossScaleFactor

        # Calculate the mean loss across all signals.
        signalReconstructedLoss[~validDataMask] = torch.nan  # Zero out the loss for invalid data points.
        signalReconstructedLoss = signalReconstructedLoss.nanmean(dim=-1)  # Dim: batchSize, numSignals
        if averageBatches: signalReconstructedLoss = signalReconstructedLoss.nanmean(dim=0)   # Dim: numSignals
        # signalReconstructedLoss: numSignals

        return signalReconstructedLoss

    # -------------------------- Loss Calculations ------------------------- #

    def calculateActivityLoss(self, predictedActivityProfile, allLabels, allLabelsMask):
        # Find the boolean flags for the data involved in the loss calculation.
        activityDataMask = self.dataInterface.getActivityColumn(allLabelsMask, self.activityLabelInd)  # Dim: batchSize
        trueActivityLabels = self.dataInterface.getActivityLabels(allLabels, allLabelsMask, self.activityLabelInd)
        batchSize, encodedDimension = predictedActivityProfile.shape
        # predictedActivityLabels: batchSize, encodedDimension
        device = predictedActivityProfile.device

        # Assert that the predicted activity profile is valid.
        assert predictedActivityProfile.ndim == 2, f"Check the predicted activity profile. Found {predictedActivityProfile.shape} shape"
        assert trueActivityLabels.ndim == 1, f"Check the true activity labels. Found {trueActivityLabels.shape} shape"

        # Apply the Gaussian weights to the predicted activity profile.
        classDimension, gaussianWeight = self.getGaussianWeights(encodedDimension, device=device, numClasses=self.numActivities)
        predictedActivityClasses = torch.zeros(batchSize, self.numActivities, device=device)

        # Apply the Gaussian weights to the predicted activity profile.
        for classInd in range(self.numActivities):
            predictedActivityClasses[:, classInd] = (predictedActivityProfile[:, classInd * classDimension:(classInd + 1) * classDimension]*gaussianWeight).sum(dim=-1)
        print(predictedActivityClasses[activityDataMask][0].round(decimals=4), trueActivityLabels[0])

        # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
        activityLosses = self.activityCrossEntropyLoss.to(device)(predictedActivityClasses[activityDataMask], trueActivityLabels.long())

        return activityLosses.nanmean()

    def calculateEmotionLoss(self, predictedEmotionProfile, allLabels, allLabelsMask):
        # Assert that the predicted activity profile is valid.
        assert predictedEmotionProfile.ndim == 3, f"Check the predicted activity profile. Found {predictedEmotionProfile.shape} shape"
        batchSize, numEmotions, encodedDimension = predictedEmotionProfile.shape
        assert self.emotionCrossEntropyLoss is not None
        device = predictedEmotionProfile.device

        # Find the boolean flags for the data involved in the loss calculation.
        emotionDataMask = self.dataInterface.getEmotionMasks(allLabelsMask, numEmotions=numEmotions)  # Dim: batchSize, numEmotions
        emotionLosses = torch.zeros(numEmotions, device=device)  # Dim: numEmotions
        # predictedEmotionProfile: batchSize, numEmotions, encodedDimension

        for emotionInd in range(numEmotions):
            # Get the true emotion labels.
            emotionMask = self.dataInterface.getEmotionColumn(emotionDataMask, emotionInd)
            trueEmotionLabels = self.dataInterface.getEmotionLabels(emotionInd, allLabels, allLabelsMask)

            # Get the true emotion labels.
            emotionProfile = predictedEmotionProfile[:, emotionInd]
            numEmotionClasses = self.allEmotionClasses[emotionInd]

            # Assert that the predicted profile is valid.
            assert (trueEmotionLabels < numEmotionClasses).all(), f"Check the true emotion labels. Found {trueEmotionLabels} with {self.allEmotionClasses[emotionInd]} classes"

            # Apply the Gaussian weights to the predicted activity profile.
            classDimension, gaussianWeight = self.getGaussianWeights(encodedDimension, device=device, numClasses=numEmotionClasses)
            emotionClasses = torch.zeros(batchSize, numEmotionClasses, device=device)

            # Apply the Gaussian weights to the predicted activity profile.
            for classInd in range(numEmotionClasses):
                emotionClasses[:, classInd] = (emotionProfile[:, classInd * classDimension:(classInd + 1) * classDimension]*gaussianWeight).sum(dim=-1)

            # Calculate the emotion classification accuracy.
            emotionLosses[emotionInd] = self.emotionCrossEntropyLoss[emotionInd].to(device)(emotionClasses[emotionMask], trueEmotionLabels.long()).nanmean()

        return emotionLosses

    def getGaussianWeights(self, encodedDimension, device, numClasses):
        classDimension = encodedDimension // numClasses

        # Generate the Gaussian weights for the predicted activity profile.
        gaussianWeight = self.gaussian_1d_kernel(classDimension, classDimension / 2, classDimension / 6, device=device)
        return classDimension, gaussianWeight

    @staticmethod
    def gaussian_1d_kernel(size, mu, std, device):
        # Create an array of indices and shift them so that the mean is at mu.
        x = torch.arange(0, size, dtype=torch.float32, device=device) - mu
        # Calculate the Gaussian function for each index.
        kernel = torch.exp(-(x ** 2) / (2 * std ** 2))
        # Normalize the kernel so that the sum of all values is 1.
        kernel /= kernel.sum()

        return kernel