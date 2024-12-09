# General
import time

import torch

# Helper classes
from .lossCalculations import lossCalculations


class organizeTrainingLosses(lossCalculations):

    def __init__(self, accelerator, allEmotionClasses, activityLabelInd):
        super(organizeTrainingLosses, self).__init__(accelerator, allEmotionClasses, activityLabelInd)

    # ---------------------------------------------------------------------- #
    # -------------------------- Loss Calculations ------------------------- #     

    def storeTrainingLosses(self, submodel, modelPipeline, lossDataLoader):
        self.accelerator.print(f"\nCalculating loss for {modelPipeline.model.datasetName} model")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask = modelPipeline.prepareInformation(lossDataLoader)
        model, allSignalData, allSignalIdentifiers, allMetadata = (tensor.to(self.accelerator.device) for tensor in (model, allSignalData, allSignalIdentifiers, allMetadata))
        # allSignalData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # allTrainingLabelMask, allTestingLabelMask: batchSize, numEmotions + 1 (activity)
        # allTrainingSignalMask, allTestingSignalMask: batchSize, numSignals
        # allSignalIdentifiers: batchSize, numSignals, numSignalIdentifiers
        # allLabels: batchSize, numEmotions + 1 (activity) + numSignals
        # allMetadata: batchSize, numMetadata

        # Stop gradient tracking.
        with torch.no_grad():
            t1 = time.time()
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = model.fullPass(submodel, allSignalData, allSignalIdentifiers, allMetadata, device=self.accelerator.device, profileTraining=False)
            t2 = time.time(); self.accelerator.print("\tFull Pass", t2 - t1)

            # Calculate the signal encoding loss.
            signalReconstructedTrainingLosses = self.calculateSignalEncodingLoss(allSignalData, reconstructedSignalData, validDataMask, allTrainingSignalMask)
            signalReconstructedTestingLosses = self.calculateSignalEncodingLoss(allSignalData, reconstructedSignalData, validDataMask, allTestingSignalMask)

            # Store the signal encoder loss information.
            self.storeLossInformation(signalReconstructedTrainingLosses, signalReconstructedTestingLosses, model.specificSignalEncoderModel.trainingLosses_signalReconstruction, model.specificSignalEncoderModel.testingLosses_signalReconstruction)
            self.accelerator.print("Reconstruction loss values:", signalReconstructedTrainingLosses.nanmean().item(), signalReconstructedTestingLosses.nanmean().item())

            # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
            # activityTestingLoss = self.calculateActivityLoss(allActivityDistributions, allLabels, allTestingMasks, activityClassWeights)
            # activityTrainingLoss = self.calculateActivityLoss(allActivityDistributions, allLabels, allTrainingMasks, activityClassWeights)
            # # Store the activity loss information.
            # self.testingLosses_activities.append(activityTestingLoss)
            # self.trainingLosses_activities.append(activityTrainingLoss)

            # # For each emotion we are predicting.
            # for emotionInd in range(self.numEmotions):
            #     if allEmotionClassWeights[emotionInd] is torch.nan: continue
            #     # Calculate and add the loss due to misclassifying an emotion.
            #     emotionTestingLoss = self.calculateEmotionLoss(emotionInd, allFinalEmotionDistributions, allLabels, allTestingMasks, allEmotionClassWeights) # Calculate the error in the emotion predictions
            #     emotionTrainingLoss = self.calculateEmotionLoss(emotionInd, allFinalEmotionDistributions, allLabels, allTrainingMasks, allEmotionClassWeights) # Calculate the error in the emotion predictions
            #     # Store the loss information.
            #     self.testingLosses_emotions[emotionInd].append(emotionTestingLoss)
            #     self.trainingLosses_emotions[emotionInd].append(emotionTrainingLoss)
            # return None

            # # For each time window analysis.
            # for timeWindowInd in range(len(modelConstants.timeWindows)):
            #     timeWindow = modelConstants.timeWindows[timeWindowInd]
            #     # If we are debugging and do not need to store all the loss values.
            #     if fastPass and timeWindow != self.generalTimeWindow: continue
            #
            #     # Segment the data into its time window.
            #     segmentedSignalData = dataAugmentation.getRecentSignalPoints(allSignalData, timeWindow)
            #
            #     t1 = time.time()
            #     # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            #     signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs = model.fullDataPass(submodel, lossDataLoader, timeWindow=timeWindow, reconstructSignals=True, compileVariables=False, trainingFlag=False)
            #     t2 = time.time(); self.accelerator.print("Full Pass", t2 - t1),
            #
            #     # Unpack all the data.
            #     segmentedCompressedData, segmentedReconstructedEncodedData, segmentedDenoisedDoubleReconstructedData, segmentedAutoencoderLayerLoss = autoencodingOutputs
            #     segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss = signalEncodingOutputs
            #     segmentedMappedSignalData, segmentedReconstructedCompressedData, segmentedFeatureData, segmentedActivityDistributions, segmentedBasicEmotionDistributions, segmentedFinalEmotionDistributions = emotionModelOutputs
            #
            #     if submodel == modelConstants.signalEncoderModel:
            #         # Calculate the signal encoding loss.
            #         signalReconstructedTestingLoss, encodedMeanTestingLoss, encodedMinMaxTestingLoss, positionalEncodingTestingLoss, decodedPositionalEncodingTestingLoss, signalEncodingTestingLayerLoss = \
            #             self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss, allTestingMasks, reconstructionIndex)
            #         signalReconstructedTrainingLoss, encodedMeanTrainingLoss, encodedMinMaxTrainingLoss, positionalEncodingTrainingLoss, decodedPositionalEncodingTrainingLoss, signalEncodingTrainingLayerLoss = \
            #             self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss, allTrainingMasks, reconstructionIndex)
            #
            #         # Store the signal encoder loss information.
            #         self.storeLossInformation(decodedPositionalEncodingTrainingLoss, decodedPositionalEncodingTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeDecodedPosEncAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeDecodedPosEncAnalysis[timeWindowInd])
            #         self.storeLossInformation(signalReconstructedTrainingLoss, signalReconstructedTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])
            #         self.storeLossInformation(encodedMinMaxTrainingLoss, encodedMinMaxTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeMinMaxAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeMinMaxAnalysis[timeWindowInd])
            #         self.storeLossInformation(positionalEncodingTrainingLoss, positionalEncodingTestingLoss, model.specificSignalEncoderModel.trainingLosses_timePosEncAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timePosEncAnalysis[timeWindowInd])
            #         self.storeLossInformation(encodedMeanTrainingLoss, encodedMeanTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeMeanAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeMeanAnalysis[timeWindowInd])
            #         # Calculate and Store the optimal loss only once.
            #         if len(model.specificSignalEncoderModel.trainingLosses_timeReconstructionOptimalAnalysis[timeWindowInd]) == 0:
            #             optimalTrainingLoss = self.getOptimalLoss(model.specificSignalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTrainingMasks, reconstructionIndex)
            #             optimalTestingLoss = self.getOptimalLoss(model.specificSignalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTestingMasks, reconstructionIndex)
            #
            #             # Store the signal encoder loss information.
            #             self.storeLossInformation(optimalTrainingLoss, optimalTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeReconstructionOptimalAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeReconstructionOptimalAnalysis[timeWindowInd])
            #         # Inform the user about the final loss.
            #         print(f"\tSignal encoder {timeWindow} second losses:", signalReconstructedTrainingLoss.item(), signalReconstructedTestingLoss.item())

    # --------------------------- Helper Methods --------------------------- #

    def storeLossInformation(self, trainingLoss, testingLoss, trainingHolder, testingHolder):
        # Gather the losses from each device (for distributed training). NOTE: don't use gather as it doesn't remove duplicates.
        gatheredTrainingLoss, gatheredTestingLoss = self.accelerator.gather_for_metrics((trainingLoss, testingLoss))

        # Store the loss information.
        trainingHolder.append(gatheredTrainingLoss.detach().cpu().numpy())
        testingHolder.append(gatheredTestingLoss.detach().cpu().numpy())

    # ----------------------- Class Weighting Methods ---------------------- #

    def getClassWeights(self, allLabels, allTrainingMasks, allTestingMasks, numActivities):
        # Initialize placeholder class weights for each emotion as nan.
        validLabelsMask = allTrainingMasks | allTestingMasks
        allEmotionClassWeights = []

        # For each emotion.
        for emotionInd in range(self.numEmotions):
            numEmotionClasses = self.allEmotionClasses[emotionInd]  # The number of classes (intensity labels) per emotion.

            # Calculate the weight/significance of each emotion class.
            validEmotionLabels = self.dataInterface.getEmotionLabels(emotionInd, allLabels, validLabelsMask)
            emotionClassWeight = self.assignClassWeights(validEmotionLabels, numEmotionClasses)
            allEmotionClassWeights.append(emotionClassWeight)

        # Assign the class weights for the activities.      
        activityLabels = self.dataInterface.getActivityLabels(allLabels, validLabelsMask, self.activityLabelInd)
        activityClassWeights = self.assignClassWeights(activityLabels, numActivities)

        return allEmotionClassWeights, activityClassWeights

    @staticmethod
    def assignClassWeights(class_labels, num_classes):
        # Remove the invalid labels.
        validLabels = ~torch.isnan(class_labels)
        validClassLabels = class_labels[validLabels]

        # Assert that the class labels are valid.
        if len(validClassLabels) == 0: return torch.nan
        assert 0 <= validClassLabels.min(), "The class labels must be non-negative."
        assert class_labels.ndim == 1, "The class labels must be 1D."

        # Initialize the class counts.
        lowerBoundLabels = validClassLabels.floor()
        upperBoundLabels = validClassLabels.ceil()

        # Get the upper and lower class weights.
        upperBoundWeight = validClassLabels - lowerBoundLabels
        lowerBoundWeight = 1 - upperBoundWeight

        # Remove invalid lower and upper bounds.
        upperBoundLabels[num_classes <= upperBoundLabels] = num_classes - 1
        lowerBoundLabels[lowerBoundLabels < 0] = 0

        # Count the number of points in each class.
        classCounts = torch.zeros(num_classes, device=class_labels.device, dtype=torch.float32)
        classCounts[lowerBoundLabels.long()] += lowerBoundWeight
        classCounts[upperBoundLabels.long()] += upperBoundWeight

        # Convert to class weights.
        class_weights = 1 / classCounts
        class_weights[torch.isnan(class_weights)] = 0

        # Set the class weights to 0 for classes with no points.
        class_weights = torch.softmax(class_weights, dim=0)

        return class_weights
