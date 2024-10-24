# General
import math

import torch

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
# Loss methods
from helperFiles.machineLearning.modelControl.Models.pyTorch.lossFunctions import pytorchLossMethods, weightLoss
from ..emotionDataInterface import emotionDataInterface
from ..generalMethods.generalMethods import generalMethods
from ..modelConstants import modelConstants
from ..modelParameters import modelParameters


class lossCalculations:

    def __init__(self, accelerator, allEmotionClasses, activityLabelInd):
        # General parameters
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: numEmotions
        self.activityLabelInd = activityLabelInd  # The index of the activity label in the label tensor.
        self.numEmotions = len(allEmotionClasses)  # The number of emotions to predict.
        self.accelerator = accelerator  # Hugging face model optimizations.

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()
        self.modelParameters = modelParameters
        self.generalMethods = generalMethods()
        self.modelHelpers = modelHelpers()

        # Specify the model's loss functions (READ BEFORE USING!!). 
        #       Classification Options: "NLLLoss", "KLDivLoss", "CrossEntropyLoss", "BCEWithLogitsLoss"
        #       Custom Classification Options: "weightedKLDiv", "diceLoss", "FocalLoss"
        #       Regression Options: "MeanSquaredError", "MeanAbsoluteError", "Huber", "SmoothL1Loss", "PoissonNLLLoss", "GammaNLLLoss"
        #       Custom Regression Options: "R2", "pearson", "LogCoshLoss", "weightedMSE"
        self.emotionDist_lossType = "MeanSquaredError"  # The loss enforcing correct distribution shape.
        self.activityClass_lossType = "CrossEntropyLoss"  # The loss enforcing correct activity recognition.
        # Initialize the loss function WITHOUT the class weights.
        self.activityClassificationLoss = pytorchLossMethods(lossType=self.activityClass_lossType, class_weights=None).loss_fn
        self.emotionClassificationLoss = pytorchLossMethods(lossType=self.emotionDist_lossType, class_weights=None).loss_fn
        self.positionalEncoderLoss = pytorchLossMethods(lossType="MeanSquaredError", class_weights=None).loss_fn
        self.reconstructionLoss = pytorchLossMethods(lossType="MeanSquaredError", class_weights=None).loss_fn

    # ---------------------------------------------------------------------- #
    # -------------------------- Loss Calculations ------------------------- #

    @staticmethod
    def getData(data, mask):
        if data is None or mask is None: return data
        return data[mask]

    def getReconstructionDataMask(self, allLabelsMask, reconstructionIndex):
        # Find the boolean flags for the data involved in the loss calculation.
        if allLabelsMask is not None and reconstructionIndex is not None:
            return self.dataInterface.getEmotionColumn(allLabelsMask, reconstructionIndex)  # Dim: numExperiments

    def getOptimalLoss(self, method, allInputData, allLabelsMask, reconstructionIndex):
        # Isolate the signals for this loss (For example, training vs. testing).
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        signalData = self.getData(allInputData, reconstructionDataMask)  # Dim: numExperiments, numSignals, signalLength

        # Calculate the final losses.
        finalLoss = method(signalData).mean()
        return finalLoss

    def calculateSignalEncodingLoss(self, allInitialSignalData, allReconstructedSignalData, physiologicalTimes, allMissingDataMask, allLabelsMask, reconstructionIndex):
        # Find the boolean flags for the data involved in the loss calculation.
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        # reconstructionDataMask dimension: numExperiments

        # Assert the validity of the input parameters.
        assert physiologicalTimes.size(-1) == allReconstructedSignalData.size(-1), "The physiological times and reconstructed signal data must have the same dimensions."

        # Unpack the signal data.
        allDatapoints = emotionDataInterface.getChannelData(allInitialSignalData, channelName=modelConstants.signalChannel)
        allTimepoints = emotionDataInterface.getChannelData(allInitialSignalData, channelName=modelConstants.timeChannel)
        # allDatapoints and allTimepoints: numExperiments, numSignals, maxSequenceLength

        # Isolate the signals for this loss (For example, training vs. testing).
        reconstructedSignalData = self.getData(allReconstructedSignalData, reconstructionDataMask)  # Dim: numExperiments, numSignals, encodedDimension
        missingDataMask = self.getData(allMissingDataMask, reconstructionDataMask)  # Dim: numExperiments, numSignals, maxSequenceLength
        datapoints = self.getData(allDatapoints, reconstructionDataMask)  # Dim: numExperiments, numSignals, maxSequenceLength
        timepoints = self.getData(allTimepoints, reconstructionDataMask)  # Dim: numExperiments, numSignals, maxSequenceLength
        batchSize, numSignals, maxSequenceLength = missingDataMask.size()
        encodedDimension = reconstructedSignalData.size(2)
        if batchSize == 0: return None

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
        closestPhysiologicalDataRight = torch.gather(input=reconstructedSignalData, dim=2, index=validIndsRight)  # Initialize the tensor.
        closestPhysiologicalDataLeft = torch.gather(input=reconstructedSignalData, dim=2, index=validIndsLeft)  # Initialize the tensor.
        assert ((closestPhysiologicalTimesLeft <= timepoints) & (timepoints <= closestPhysiologicalTimesRight)).all(), "The timepoints must be within the range of the closest physiological times."
        # closestPhysiologicalData dimension: batchSize, numSignals, maxSequenceLength

        # Perform linear interpolation.
        linearSlopes = (closestPhysiologicalDataRight - closestPhysiologicalDataLeft) / (closestPhysiologicalTimesRight - closestPhysiologicalTimesLeft).clamp(min=1e-4)
        linearSlopes[closestPhysiologicalTimesLeft == closestPhysiologicalTimesRight] = 0

        # Calculate the error in signal reconstruction (encoding loss).
        interpolatedData = closestPhysiologicalDataLeft + (timepoints - closestPhysiologicalTimesLeft) * linearSlopes
        physiologicalDataUncertainty = (closestPhysiologicalDataRight - closestPhysiologicalDataLeft).pow(2)  # Initialize the uncertainty tensor.
        signalReconstructedLoss = (datapoints - interpolatedData).pow(2)
        # signalReconstructedLoss dimension: batchSize, numSignals, maxSequenceLength

        # Mask out the missing data.
        validDataMask = ~missingDataMask & (physiologicalDataUncertainty <= signalReconstructedLoss)
        signalReconstructedLoss = signalReconstructedLoss[validDataMask].mean()

        # Assert that nothing is wrong with the loss calculations.
        self.modelHelpers.assertVariableIntegrity(signalReconstructedLoss, variableName="encoded signal reconstructed loss", assertGradient=False)

        return signalReconstructedLoss

    def calculateActivityLoss(self, predictedActivityLabels, allLabels, allLabelsMask, activityClassWeights):
        # Find the boolean flags for the data involved in the loss calculation.
        activityDataMask = self.dataInterface.getActivityColumn(allLabelsMask, self.activityLabelInd)  # Dim: numExperiments
        trueActivityLabels = self.dataInterface.getActivityLabels(allLabels, allLabelsMask, self.activityLabelInd)

        # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
        activityLosses = self.activityClassificationLoss(predictedActivityLabels[activityDataMask], trueActivityLabels.long())
        activityLoss = weightLoss(activityLosses, activityClassWeights, trueActivityLabels)
        assert not activityLoss.isnan().any().item() and not activityLoss.isinf().any().item(), f"Check your inputs to (or the method) self.activityClassificationLoss. Found {activityLoss} value"

        return activityLoss

    def calculateEmotionsLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
        # Calculate the loss from predicting similar basic emotions.
        emotionOrthogonalityLoss = self.lossCalculations.scoreEmotionOrthonormality(allBasicEmotionDistributions)
        assert not emotionOrthogonalityLoss.isnan().any().item() and not emotionOrthogonalityLoss.isinf().any().item()
        # Calculate the loss from model-specific weights.
        modelSpecificWeights = self.lossCalculations.scoreModelWeights(self.model.predictUserEmotions.allSubjectWeights)
        assert not modelSpecificWeights.isnan().any().item() and not modelSpecificWeights.isinf().any().item()
        # Add all the losses together into one value.
        finalLoss = reconstructedLoss + activityLoss * 2 + emotionOrthogonalityLoss / 2  # + modelSpecificWeights*0.01

        # Get the valid emotion indices (ones with training points).
        batchEmotionTrainingMask = self.dataInterface.getEmotionMasks(batchTrainingMask)
        validEmotionInds = self.dataInterface.getLabelInds_withPoints(batchEmotionTrainingMask)

        emotionLoss = 0
        # For each emotion we are predicting that has training data.
        for validEmotionInd in validEmotionInds:
            # Calculate and add the loss due to misclassifying the emotion.
            emotionLoss = self.calculateEmotionLoss(validEmotionInd, predictedBatchEmotions, trueBatchLabels,
                                                                     batchTrainingMask, allEmotionClassWeights)  # Calculate the error in the emotion predictions
            emotionLoss += emotionLoss / len(validEmotionInds)  # Add all the losses together into one value.
        # Average all the losses that were added together.
        finalLoss += emotionLoss * 2

    def calculateEmotionLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
        # Organize the emotion's training information.
        emotionLabels = self.dataInterface.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
        emotionClassWeights = allEmotionClassWeights[emotionInd]

        # Get the predicted and true emotion distributions.
        predictedTrainingEmotions, trueTrainingEmotions = self.dataInterface.getEmotionDistributions(emotionInd, predictedEmotionlabels, allLabels, allLabelsMask)
        # predictedTrainingEmotions = F.normalize(predictedTrainingEmotions, dim=1, p=1)
        # assert (predictedTrainingEmotions >= 0).all()

        # Calculate an array of possible emotion ratings.
        numEmotionClasses = self.allEmotionClasses[emotionInd]
        possibleEmotionRatings = torch.arange(0, numEmotionClasses, numEmotionClasses / self.emotionLength, device=allLabels.mainDevice) - 0.5
        # Calculate the weighted prediction losses
        mseLossDistributions = (emotionLabels[:, None] - possibleEmotionRatings) ** 2
        emotionDistributionLosses = (mseLossDistributions * predictedTrainingEmotions).sum(dim=1)

        # Calculate the error in the emotion predictions
        # emotionDistributionLosses = self.emotionClassificationLoss(predictedTrainingEmotions, trueTrainingEmotions.float()).sum(dim=-1)
        emotionDistributionLoss = weightLoss(emotionDistributionLosses, emotionClassWeights, emotionLabels).mean()
        assert not emotionDistributionLoss.isnan().any().item() and not emotionDistributionLoss.isinf().any().item(), print(predictedTrainingEmotions, trueTrainingEmotions.float(), emotionDistributionLoss)

        return emotionDistributionLoss

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
        probabilityOverlap_basicEmotions -= torch.eye(numBasicEmotion, numBasicEmotion, device=allBasicEmotionDistributions.mainDevice).view(1, 1, numBasicEmotion, numBasicEmotion)
        # For each interpretation of emotions, the basis states should be orthonormal.
        basicEmotion_orthoganalityLoss = probabilityOverlap_basicEmotions.mean()

        # Calculate the overlap in probability for each basic emotion across each interpretation.
        allInterpretationEmotions = allBasicEmotionDistributionsAbs.permute(0, 2, 1, 3)  # batchSize, numBasicEmotions, numInterpreterHeads, emotionLength
        allInterpretationEmotions_T = allBasicEmotionDistributionsAbs.permute(0, 2, 3, 1)  # batchSize, numBasicEmotions, emotionLength, numInterpreterHeads
        probabilityOverlap_interpretations = allInterpretationEmotions.sqrt() @ allInterpretationEmotions_T.sqrt()
        # Zero out self-overlap as each signal SHOULD be overlapping with itself.
        probabilityOverlap_interpretations -= torch.eye(numInterpreterHeads, numInterpreterHeads, device=allBasicEmotionDistributions.mainDevice).view(1, 1, numInterpreterHeads, numInterpreterHeads)
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

    def errorPerClass(self, output, target):
        with torch.no_grad():
            # Calculate the error per class
            batchSize, num_classes = output.size()
            class_errors = torch.zeros(num_classes)
            uniqueClasses = torch.unique(target[0], return_inverse=False, return_counts=False, dim=-1, sorted=True)

            for i in range(len(uniqueClasses)):
                # Mask for the current class
                class_mask = (target[0] == uniqueClasses[i])

                if True in class_mask:
                    class_errors[i] = self.positionalEncoderLoss(output[:, class_mask], target[:, class_mask]).mean() * (num_classes-1)**2

            print("Loss per class:", class_errors)
            print("Final classes:", output[0:2])

    @staticmethod
    def gradient_penalty(inputs, outputs, dims):
        # Calculate the gradient wrt the inputs.
        gradients = torch.autograd.grad(
            grad_outputs=torch.ones_like(outputs, device=outputs.mainDevice),
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

    def calculateMinMaxLoss(self, inputData, expectedMean=0, expectedMinMax=1, dim=-1, minMaxBuffer=0.0):
        # Calculate the min-max loss.
        minMaxData = self.generalMethods.minMaxScale_noInverse(inputData, scale=expectedMinMax, buffer=minMaxBuffer)
        minMaxLoss = (minMaxData - expectedMean).pow(2)

        # Calculate the mean error.
        meanData = inputData.mean(dim=dim)
        meanError = (meanData - expectedMean).pow(2)

        return meanError, minMaxLoss

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
