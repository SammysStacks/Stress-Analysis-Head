import math

import torch
from torch import nn

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.lossFunctions import pytorchLossMethods
from ..emotionDataInterface import emotionDataInterface
from ..modelConstants import modelConstants
from ..modelParameters import modelParameters


class lossCalculations:

    def __init__(self, accelerator, allEmotionClasses, activityLabelInd):
        # General parameters
        self.numEmotions = len(allEmotionClasses or [])  # The number of emotions to predict.
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: numEmotions
        self.activityLabelInd = activityLabelInd  # The index of the activity label in the label tensor.
        self.accelerator = accelerator  # Hugging face model optimizations.

        # Calculate the number of sequence points to throw out.
        self.scaledLossPercent = 5  # The percentage of points to remove from the loss calculation.
        self.lossScaleFactor = 10  # The factor to scale the loss by when removing points.
        self.minRemovalPoints = 2  # The minimum number of points to remove from the loss calculation.
        self.maxRemovalPoints = 4  # The maximum number of points to remove from the loss calculation.

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()
        self.modelHelpers = modelHelpers()

        # Specify the model's loss functions (READ BEFORE USING!!).
        #       Classification Options: "NLLLoss", "KLDivLoss", "CrossEntropyLoss", "BCEWithLogitsLoss"
        #       Custom Classification Options: "weightedKLDiv", "diceLoss", "FocalLoss"
        #       Regression Options: "MeanSquaredError", "MeanAbsoluteError", "Huber", "SmoothL1Loss", "PoissonNLLLoss", "GammaNLLLoss"
        #       Custom Regression Options: "R2", "pearson", "LogCoshLoss", "weightedMSE"
        # Initialize the loss function WITHOUT the class weights.
        self.meanSquaredError = pytorchLossMethods(lossType="MeanSquaredError", class_weights=None).loss_fn
        self.smoothL1Loss = nn.SmoothL1Loss(reduction='none', beta=1)  # NEVER MAKE IT LESS THAN 1, its just MAE for outliers.

    # -------------------------- Signal Encoder Loss Calculations ------------------------- #

    def calculateSignalEncodingLoss(self, allInitialSignalData, allReconstructedSignalData, allValidDataMask, allSignalMask, averageBatches=True):
        # Get the relevant data for the loss calculation.
        allDatapoints = emotionDataInterface.getChannelData(allInitialSignalData, channelName=modelConstants.signalChannel)
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

        # Get the number of signal points to remove from the loss calculation.
        numBatchSignalPoints = validDataMask.sum(dim=-1)  # Dim: batchSize, numSignals
        numRemovalPoints = torch.ceil(self.scaledLossPercent * numBatchSignalPoints / 100)  # Dim: batchSize, numSignals
        numRemovalPoints.clamp(self.minRemovalPoints, self.maxRemovalPoints)  # Ensure that at most 4 points are removed.

        for numPointsRemoved in range(numRemovalPoints.max().int().item()):
            # Find the current most noisy points within each signal's batch.
            findMaxLoss = torch.where(validDataMask, signalReconstructedLoss, float('-inf'))
            max_indices = findMaxLoss.argmax(dim=-1, keepdim=True).squeeze(-1)

            # Each signal's batch has a different number of points to remove.
            validRemovalPointsMask = (numRemovalPoints <= numPointsRemoved)
            validSignalIndices = signal_indices[validRemovalPointsMask]
            validBatchIndices = batch_indices[validRemovalPointsMask]
            validMaxIndices = max_indices[validRemovalPointsMask]

            # Remove the top noisy points from the loss calculation.
            signalReconstructedLoss[validBatchIndices, validSignalIndices, validMaxIndices] /= self.lossScaleFactor

        # Calculate the mean loss across all signals.
        signalReconstructedLoss[~validDataMask] = torch.nan  # Zero out the loss for invalid data points.
        signalReconstructedLoss = signalReconstructedLoss.nanmean(dim=-1)  # Dim: batchSize, numSignals
        if averageBatches: signalReconstructedLoss = signalReconstructedLoss.nanmean(dim=0)   # Dim: numSignals
        # signalReconstructedLoss: numSignals

        return signalReconstructedLoss

    # -------------------------- Loss Calculations ------------------------- #

    # def calculateActivityLoss(self, predictedActivityLabels, allLabels, allLabelsMask, activityClassWeights):
    #     # Find the boolean flags for the data involved in the loss calculation.
    #     activityDataMask = self.dataInterface.getActivityColumn(allLabelsMask, self.activityLabelInd)  # Dim: numExperiments
    #     trueActivityLabels = self.dataInterface.getActivityLabels(allLabels, allLabelsMask, self.activityLabelInd)
    #
    #     # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
    #     activityLosses = self.activityClassificationLoss(predictedActivityLabels[activityDataMask], trueActivityLabels.long())
    #     activityLoss = weightLoss(activityLosses, activityClassWeights, trueActivityLabels)
    #     assert not activityLoss.isnan().any() and not activityLoss.isinf().any(), f"Check your inputs to (or the method) self.activityClassificationLoss. Found {activityLoss} value"
    #
    #     return activityLoss

    # def calculateEmotionsLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
    #     # Calculate the loss from predicting similar basic emotions.
    #     emotionOrthogonalityLoss = self.lossCalculations.scoreEmotionOrthonormality(allBasicEmotionDistributions)
    #     assert not emotionOrthogonalityLoss.isnan().any() and not emotionOrthogonalityLoss.isinf().any()
    #     # Calculate the loss from model-specific weights.
    #     modelSpecificWeights = self.lossCalculations.scoreModelWeights(self.model.predictUserEmotions.allSubjectWeights)
    #     assert not modelSpecificWeights.isnan().any() and not modelSpecificWeights.isinf().any()
    #     # Add all the losses together into one value.
    #     finalLoss = reconstructedLoss + activityLoss * 2 + emotionOrthogonalityLoss / 2  # + modelSpecificWeights*0.01
    #
    #     # Get the valid emotion indices (ones with training points).
    #     batchEmotionTrainingMask = self.dataInterface.getEmotionMasks(batchTrainingMask)
    #     validEmotionInds = self.dataInterface.getLabelInds_withPoints(batchEmotionTrainingMask)
    #
    #     emotionLoss = 0
    #     # For each emotion we are predicting that has training data.
    #     for validEmotionInd in validEmotionInds:
    #         # Calculate and add the loss due to misclassifying the emotion.
    #         emotionLoss = self.calculateEmotionLoss(validEmotionInd, predictedBatchEmotions, trueBatchLabels,
    #                                                                  batchTrainingMask, allEmotionClassWeights)  # Calculate the error in the emotion predictions
    #         emotionLoss += emotionLoss / len(validEmotionInds)  # Add all the losses together into one value.
    #     # Average all the losses that were added together.
    #     finalLoss += emotionLoss * 2
    #
    # def calculateEmotionLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
    #     # Organize the emotion's training information.
    #     emotionLabels = self.dataInterface.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
    #     emotionClassWeights = allEmotionClassWeights[emotionInd]
    #
    #     # Get the predicted and true emotion distributions.
    #     predictedTrainingEmotions, trueTrainingEmotions = self.dataInterface.getEmotionDistributions(emotionInd, predictedEmotionlabels, allLabels, allLabelsMask)
    #     # predictedTrainingEmotions = F.normalize(predictedTrainingEmotions, dim=1, p=1)
    #     # assert (predictedTrainingEmotions >= 0).all()
    #
    #     # Calculate an array of possible emotion ratings.
    #     numEmotionClasses = self.allEmotionClasses[emotionInd]
    #     possibleEmotionRatings = torch.arange(0, numEmotionClasses, numEmotionClasses / self.emotionLength, device=allLabels.device) - 0.5
    #     # Calculate the weighted prediction losses
    #     mseLossDistributions = (emotionLabels[:, None] - possibleEmotionRatings) ** 2
    #     emotionDistributionLosses = (mseLossDistributions * predictedTrainingEmotions).sum(dim=1)
    #
    #     # Calculate the error in the emotion predictions
    #     # emotionDistributionLosses = self.emotionClassificationLoss(predictedTrainingEmotions, trueTrainingEmotions.float()).sum(dim=-1)
    #     emotionDistributionLoss = weightLoss(emotionDistributionLosses, emotionClassWeights, emotionLabels).mean()
    #     assert not emotionDistributionLoss.isnan().any().item() and not emotionDistributionLoss.isinf().any().item(), print(predictedTrainingEmotions, trueTrainingEmotions.float(), emotionDistributionLoss)
    #
    #     return emotionDistributionLoss

    # ---------------------------------------------------------------------- #
    # ------------------------- Loss Helper Methods ------------------------ # 

    @staticmethod
    def scoreEmotionOrthonormality(allBasicEmotionDistributions):
        assert not allBasicEmotionDistributions.isnan().any().item() and not allBasicEmotionDistributions.isinf().any().item()
        batchSize, numInterpreterHeads, numBasicEmotion, emotionLength = allBasicEmotionDistributions.shape
        allBasicEmotionDistributionsAbs = allBasicEmotionDistributions.abs()

        # Calculate the overlap in probability between each basic emotion.
        allBasicEmotionDistributionsAbs_T = allBasicEmotionDistributionsAbs.permute(0, 1, 3, 2)  # batchSize, self.numInterpreterHeads, emotionLength, numBasicEmotions
        probabilityOverlap_basicEmotions = allBasicEmotionDistributionsAbs.sqrt() @ allBasicEmotionDistributionsAbs_T.sqrt()
        # Zero out self-overlap as each signal SHOULD be overlapping with itself.
        probabilityOverlap_basicEmotions -= torch.eye(numBasicEmotion, numBasicEmotion, device=allBasicEmotionDistributions.device).view(1, 1, numBasicEmotion, numBasicEmotion)
        # For each interpretation of emotions, the basis states should be orthonormal.
        basicEmotion_orthoganalityLoss = probabilityOverlap_basicEmotions.mean()

        # Calculate the overlap in probability for each basic emotion across each interpretation.
        allInterpretationEmotions = allBasicEmotionDistributionsAbs.permute(0, 2, 1, 3)  # batchSize, numBasicEmotions, numInterpreterHeads, emotionLength
        allInterpretationEmotions_T = allBasicEmotionDistributionsAbs.permute(0, 2, 3, 1)  # batchSize, numBasicEmotions, emotionLength, numInterpreterHeads
        probabilityOverlap_interpretations = allInterpretationEmotions.sqrt() @ allInterpretationEmotions_T.sqrt()
        # Zero out self-overlap as each signal SHOULD be overlapping with itself.
        probabilityOverlap_interpretations -= torch.eye(numInterpreterHeads, numInterpreterHeads, device=allBasicEmotionDistributions.device).view(1, 1, numInterpreterHeads, numInterpreterHeads)
        # Between all interpretations, each basis state should be different.
        emotionInterpretation_orthoganalityLoss = probabilityOverlap_basicEmotions.mean()

        return basicEmotion_orthoganalityLoss + emotionInterpretation_orthoganalityLoss

    @staticmethod
    def scoreModelWeights(allSubjectWeights):
        """
        allSubjectWeights : numSubjects, self.numInterpreterHeads, numBasicEmotions, 1
        """
        allSubjectWeights = allSubjectWeights.squeeze(3)
        # Calculate the different in how each subject interprets their emotions.
        allSubjectWeights_subjectDeviation = allSubjectWeights[None, :, :, :] - allSubjectWeights[:, None, :, :]
        # For every basic emotion, every subject should have the same interpretation (weight for each interpretation).
        subjectDeviationNorm = torch.norm(allSubjectWeights_subjectDeviation, dim=3)[0, 1:]
        subjectDeviationNormLoss = subjectDeviationNorm.mean()

        # # For every predicted emotion, the model should recombine the emotions similarly (correlation among emotions).
        # allBasicEmotionWeights = self.model.predictComplexEmotions.allBasicEmotionWeights.squeeze(3).squeeze(1)
        # allBasicEmotionWeights_Norms = torch.cdist(allBasicEmotionWeights, allBasicEmotionWeights, p=2.0)
        # allBasicEmotionWeights_Loss = allBasicEmotionWeights_Norms.mean(dim=1)
        # weightRegularizationLoss dimension: self.numEmotions, 1

        return subjectDeviationNormLoss

    # ---------------------------------------------------------------------- #
    # ----------------------- Standardization Losses ----------------------- #

    @staticmethod
    def gradient_penalty(inputs, outputs, dims):
        # Calculate the gradient wrt the inputs.
        gradients = torch.autograd.grad(
            grad_outputs=torch.ones_like(outputs, device=outputs.device),
            allow_unused=False,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
            outputs=outputs,
            inputs=inputs
        )

        # Get the size of the gradients for normalization.
        batchSize, elemA, elemB = gradients[0].size()

        # Calculate the norm of the gradients.
        gradients_norm = torch.norm(gradients[0], p='fro', dim=dims)
        gradients_norm = gradients_norm / math.sqrt(elemA*elemB)

        return gradients_norm

    @staticmethod
    def calculateStandardizationLoss(inputData, expectedMean=0, expectedStandardDeviation=1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim)
        meanData = inputData.mean(dim=dim)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = (standardDeviationData - expectedStandardDeviation).pow(2)
        meanError = (meanData - expectedMean).pow(2)

        return meanError, standardDeviationError

    def calculateDataDistributionLoss(self, originalData, predictedData, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = originalData.std(dim=dim)
        meanData = originalData.mean(dim=dim)

        meanError, standardDeviationError = self.calculateStandardizationLoss(inputData=predictedData, expectedMean=meanData, expectedStandardDeviation=standardDeviationData, dim=dim)
        return meanError, standardDeviationError

    @staticmethod
    def standardize(data, dataMean=None, dataSTD=None):
        if dataMean is None and dataSTD is None:
            dataMean = data.mean(dim=-1, keepdim=True)
            dataSTD = data.std(dim=-1, keepdim=True)

        return (data - dataMean) / (1e-10 + dataSTD), dataMean, dataSTD
