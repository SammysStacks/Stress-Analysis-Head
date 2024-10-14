# General
import time

import torch

# Helper classes
from .lossCalculations import lossCalculations
from ..generalMethods.dataAugmentation import dataAugmentation
from ..modelConstants import modelConstants


class organizeTrainingLosses(lossCalculations):

    def __init__(self, accelerator, model, allEmotionClasses, activityLabelInd, generalTimeWindow, useFinalParams=False):
        super(organizeTrainingLosses, self).__init__(accelerator, model, allEmotionClasses, activityLabelInd, useFinalParams)
        self.generalTimeWindow = generalTimeWindow

    # ---------------------------------------------------------------------- #
    # -------------------------- Loss Calculations ------------------------- #     

    def storeTrainingLosses(self, submodel, modelPipeline, lossDataLoader, fastPass):
        self.accelerator.print(f"\nCalculating loss for {modelPipeline.model.datasetName} model", flush=True)

        # Prepare the model/data for evaluation.
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = lossDataLoader.dataset.getAll()
        allSignalData, allSignalIdentifiers, allMetadata = self.dataInterface.separateData(allData)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        # Stop gradient tracking.
        with torch.no_grad():

            if submodel == modelConstants.emotionModel:
                # Segment the data into its time window.
                segmentedSignalData = dataAugmentation.getRecentSignalPoints(allSignalData, self.generalTimeWindow)

                t1 = time.time()
                # Pass all the data through the model and store the emotions, activity, and intermediate variables.
                signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs = model.fullDataPass(submodel, lossDataLoader, timeWindow=self.generalTimeWindow, reconstructSignals=False, compileVariables=False, trainingFlag=False)
                t2 = time.time(); self.accelerator.print("Full Pass", t2 - t1),

                # Unpack all the data.
                allCompressedData, allReconstructedEncodedData, allDenoisedDoubleReconstructedData, allAutoencoderLayerLoss = autoencodingOutputs
                allEncodedData, allReconstructedData, allPredictedIndexProbabilities, allDecodedPredictedIndexProbabilities, allSignalEncodingLayerLoss = signalEncodingOutputs
                allMappedSignalData, allReconstructedCompressedData, allFeatureData, allActivityDistributions, allBasicEmotionDistributions, allFinalEmotionDistributions = emotionModelOutputs

                # Calculate the signal encoding loss.
                # manifoldReconstructedTestingLoss, manifoldMeanTestingLoss, manifoldMinMaxTestingLoss = \
                #     self.calculateManifoldReductionLoss(allEncodedData, allManifoldData, allTransformedManifoldData, allReconstructedEncodedData, allTestingMasks, reconstructionIndex)
                # manifoldReconstructedTrainingLoss, manifoldMeanTrainingLoss, manifoldMinMaxTrainingLoss = \
                #     self.calculateManifoldReductionLoss(allEncodedData, allManifoldData, allTransformedManifoldData, allReconstructedEncodedData, allTrainingMasks, reconstructionIndex)
                # # Store the latent reconstruction loss.
                # self.storeLossInformation(manifoldReconstructedTrainingLoss, manifoldReconstructedTestingLoss, model.signalMappingModel.trainingLosses_encodingReconstruction,
                #                           model.signalMappingModel.testingLosses_encodingReconstruction)
                # self.storeLossInformation(manifoldMinMaxTrainingLoss, manifoldMinMaxTestingLoss, model.signalMappingModel.trainingLosses_manifoldMinMax, model.signalMappingModel.testingLosses_manifoldMinMax)
                # self.storeLossInformation(manifoldMeanTrainingLoss, manifoldMeanTestingLoss, model.signalMappingModel.trainingLosses_manifoldMean, model.signalMappingModel.testingLosses_manifoldMean)

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
                return None

            # For each time window analysis.
            for timeWindowInd in range(len(modelConstants.timeWindows)):
                timeWindow = modelConstants.timeWindows[timeWindowInd]
                # If we are debugging and do not need to store all the loss values.
                if fastPass and timeWindow != self.generalTimeWindow: continue

                # Segment the data into its time window.
                segmentedSignalData = dataAugmentation.getRecentSignalPoints(allSignalData, timeWindow)

                t1 = time.time()
                # Pass all the data through the model and store the emotions, activity, and intermediate variables.
                signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs = model.fullDataPass(submodel, lossDataLoader, timeWindow=timeWindow, reconstructSignals=True, compileVariables=False, trainingFlag=False)
                t2 = time.time(); self.accelerator.print("Full Pass", t2 - t1),

                # Unpack all the data.
                segmentedCompressedData, segmentedReconstructedEncodedData, segmentedDenoisedDoubleReconstructedData, segmentedAutoencoderLayerLoss = autoencodingOutputs
                segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss = signalEncodingOutputs
                segmentedMappedSignalData, segmentedReconstructedCompressedData, segmentedFeatureData, segmentedActivityDistributions, segmentedBasicEmotionDistributions, segmentedFinalEmotionDistributions = emotionModelOutputs

                if submodel == modelConstants.signalEncoderModel:
                    # Calculate the signal encoding loss.
                    signalReconstructedTestingLoss, encodedMeanTestingLoss, encodedMinMaxTestingLoss, positionalEncodingTestingLoss, decodedPositionalEncodingTestingLoss, signalEncodingTestingLayerLoss = \
                        self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss, allTestingMasks, reconstructionIndex)
                    signalReconstructedTrainingLoss, encodedMeanTrainingLoss, encodedMinMaxTrainingLoss, positionalEncodingTrainingLoss, decodedPositionalEncodingTrainingLoss, signalEncodingTrainingLayerLoss = \
                        self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedPredictedIndexProbabilities, segmentedDecodedPredictedIndexProbabilities, segmentedSignalEncodingLayerLoss, allTrainingMasks, reconstructionIndex)

                    # Store the signal encoder loss information.
                    self.storeLossInformation(decodedPositionalEncodingTrainingLoss, decodedPositionalEncodingTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeDecodedPosEncAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeDecodedPosEncAnalysis[timeWindowInd])
                    self.storeLossInformation(signalReconstructedTrainingLoss, signalReconstructedTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])
                    self.storeLossInformation(encodedMinMaxTrainingLoss, encodedMinMaxTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeMinMaxAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeMinMaxAnalysis[timeWindowInd])
                    self.storeLossInformation(positionalEncodingTrainingLoss, positionalEncodingTestingLoss, model.specificSignalEncoderModel.trainingLosses_timePosEncAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timePosEncAnalysis[timeWindowInd])
                    self.storeLossInformation(encodedMeanTrainingLoss, encodedMeanTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeMeanAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeMeanAnalysis[timeWindowInd])
                    # Calculate and Store the optimal loss only once.
                    if len(model.specificSignalEncoderModel.trainingLosses_timeReconstructionOptimalAnalysis[timeWindowInd]) == 0:
                        optimalTrainingLoss = self.getOptimalLoss(model.specificSignalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTrainingMasks, reconstructionIndex)
                        optimalTestingLoss = self.getOptimalLoss(model.specificSignalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTestingMasks, reconstructionIndex)

                        # Store the signal encoder loss information.
                        self.storeLossInformation(optimalTrainingLoss, optimalTestingLoss, model.specificSignalEncoderModel.trainingLosses_timeReconstructionOptimalAnalysis[timeWindowInd], model.specificSignalEncoderModel.testingLosses_timeReconstructionOptimalAnalysis[timeWindowInd])
                    # Inform the user about the final loss.
                    print(f"\tSignal encoder {timeWindow} second losses:", signalReconstructedTrainingLoss.item(), signalReconstructedTestingLoss.item())

    # --------------------------- Helper Methods --------------------------- #

    def storeLossInformation(self, trainingLoss, testingLoss, trainingHolder, testingHolder):
        # Gather the losses from each device (for distributed training). NOTE: don't use gather as it doesn't remove duplicates.
        gatheredTrainingLoss, gatheredTestingLoss = self.accelerator.gather_for_metrics((trainingLoss, testingLoss))

        # Store the loss information.
        trainingHolder.append(gatheredTrainingLoss.detach().mean().item())
        testingHolder.append(gatheredTestingLoss.detach().mean().item())

    # ----------------------- Class Weighting Methods ---------------------- #

    def getClassWeights(self, allLabels, allTrainingMasks, allTestingMasks, numActivities):
        # Initialize placeholder class weights for each emotion as nan.
        allEmotionClassWeights = [torch.nan for _ in range(self.numEmotions)]

        # Get the valid emotion indices (ones with training points).
        emotionTrainingMasks = self.dataInterface.getEmotionMasks(allTrainingMasks, self.numEmotions)
        validEmotionInds = self.dataInterface.getLabelInds_withPoints(emotionTrainingMasks)
        # Create a boolean flag to find any good point.
        validLabelsMask = allTrainingMasks | allTestingMasks

        # For each emotion that has training data.
        for validEmotionInd in validEmotionInds:
            numEmotionClasses = self.allEmotionClasses[validEmotionInd]  # The number of classes (intensity labels) per emotion.

            # Calculate the weight/significance of each emotion class.
            validEmotionLabels = self.dataInterface.getEmotionLabels(validEmotionInd, allLabels, validLabelsMask)
            emotionClassWeight = self.assignClassWeights(validEmotionLabels, numEmotionClasses)
            allEmotionClassWeights[validEmotionInd] = emotionClassWeight

        # Assign the class weights for the activities.      
        activityLabels = self.dataInterface.getActivityLabels(allLabels, validLabelsMask, self.activityLabelInd)
        activityClassWeights = self.assignClassWeights(activityLabels, numActivities)

        return allEmotionClassWeights, activityClassWeights

    @staticmethod
    def assignClassWeights(class_labels, num_classes):
        # Initialize the class counts to zero
        class_counts = torch.zeros(num_classes, device=class_labels.device)

        # Accumulate counts for each class
        for label in class_labels:
            if torch.isnan(label): continue

            lower_class = int(label)
            # Distribute weight for classes around the floating point
            if lower_class == label:
                # If the label is an integer, it gets full weight
                class_counts[lower_class] += 1
            else:
                # If the label is not an integer, distribute the weight
                upper_class = lower_class + 1
                lower_weight = upper_class - label
                upper_weight = label - lower_class
                class_counts[lower_class] += lower_weight
                class_counts[upper_class] += upper_weight

        # Handle classes with zero counts to avoid division by zero
        nonzero_counts_mask = class_counts > 0
        # Calculate weights for each class
        class_weights = torch.zeros_like(class_counts, device=class_labels.device)
        class_weights[nonzero_counts_mask] = 1.0 / class_counts[nonzero_counts_mask]

        # Normalize weights to sum to 1
        class_weights /= class_weights.sum()

        return class_weights
