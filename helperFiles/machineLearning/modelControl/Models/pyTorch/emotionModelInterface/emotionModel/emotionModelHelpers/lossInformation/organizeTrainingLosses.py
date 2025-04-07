# General
import time

import numpy as np
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.lossInformation.lossCalculations import lossCalculations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class organizeTrainingLosses(lossCalculations):

    def __init__(self, accelerator, allEmotionClasses, numActivities, activityLabelInd):
        super(organizeTrainingLosses, self).__init__(accelerator, allEmotionClasses, numActivities, activityLabelInd)

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
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile = model.fullPass(submodel, allSignalData, allSignalIdentifiers, allMetadata, device=self.accelerator.device, profileEpoch=None)
            t2 = time.time(); self.accelerator.print("\tFull Pass", t2 - t1)

            if submodel == modelConstants.signalEncoderModel:
                # Calculate the signal encoding loss.
                signalReconstructedTrainingLosses = self.calculateSignalEncodingLoss(allSignalData, reconstructedSignalData, validDataMask, allTrainingSignalMask, averageBatches=True)
                signalReconstructedTestingLosses = self.calculateSignalEncodingLoss(allSignalData, reconstructedSignalData, validDataMask, allTestingSignalMask, averageBatches=True)
                # signalReconstructedTrainingLosses: numTrainingSignals
                # signalReconstructedTestingLosses: numTestingSignals

                # Get the encoder information.
                givensAnglesPath, normalizationFactorsPath, givensAnglesFeaturesPath, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams(submodelString="SignalEncoderModel")
                activationParamsPath, moduleNames = model.getActivationParamsFullPassPath(submodelString="SignalEncoderModel")
                numFreeParamsPath, _, _ = model.getFreeParamsFullPassPath(submodelString="SignalEncoderModel")
                # givensAnglesFeaturesPath: numModuleLayers, numFeatures=5, numValues*
                # normalizationFactorsPath: numModuleLayers, numSignals, numParams=1
                # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                # givensAnglesPath: numModuleLayers, numSignals, numParams
                # activationParamsPath: numActivations, numParams=3

                # Store the signal encoder loss information.
                self.storeLossInformation(trainingLoss=signalReconstructedTrainingLosses, testingLoss=signalReconstructedTestingLosses, trainingHolder=model.specificSignalEncoderModel.trainingLosses_signalReconstruction, testingHolder=model.specificSignalEncoderModel.testingLosses_signalReconstruction)
                self.storeLossInformation(trainingLoss=activationParamsPath, testingLoss=None, trainingHolder=model.specificSignalEncoderModel.activationParamsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=normalizationFactorsPath, testingLoss=None, trainingHolder=model.specificSignalEncoderModel.normalizationFactorsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=givensAnglesFeaturesPath, testingLoss=None, trainingHolder=model.specificSignalEncoderModel.givensAnglesFeaturesPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=numFreeParamsPath, testingLoss=None, trainingHolder=model.specificSignalEncoderModel.numFreeParams, testingHolder=None)
                self.accelerator.print("Reconstruction loss values:", signalReconstructedTrainingLosses.nanmean().item(), signalReconstructedTestingLosses.nanmean().item())

            elif submodel == modelConstants.emotionModel:
                # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
                activityTestingLoss = self.calculateActivityLoss(activityProfile, allLabels, allTestingLabelMask)
                activityTrainingLoss = self.calculateActivityLoss(activityProfile, allLabels, allTrainingLabelMask)

                # Get the encoder information.
                givensAnglesPath, normalizationFactorsPath, givensAnglesFeaturesPath, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams(submodelString="ActivityModel")
                activationParamsPath, moduleNames = model.getActivationParamsFullPassPath(submodelString="ActivityModel")
                numFreeParamsPath, _, _ = model.getFreeParamsFullPassPath(submodelString="ActivityModel")
                # givensAnglesFeaturesPath: numModuleLayers, numFeatures=5, numValues*
                # normalizationFactorsPath: numModuleLayers, numSignals, numParams=1
                # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                # givensAnglesPath: numModuleLayers, numSignals, numParams
                # activationParamsPath: numActivations, numParams=3

                # Store the signal encoder loss information.
                self.storeLossInformation(trainingLoss=activityTrainingLoss, testingLoss=activityTestingLoss, trainingHolder=model.specificActivityModel.trainingLosses_signalReconstruction, testingHolder=model.specificActivityModel.testingLosses_signalReconstruction)
                self.storeLossInformation(trainingLoss=activationParamsPath, testingLoss=None, trainingHolder=model.specificActivityModel.activationParamsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=normalizationFactorsPath, testingLoss=None, trainingHolder=model.specificActivityModel.normalizationFactorsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=givensAnglesFeaturesPath, testingLoss=None, trainingHolder=model.specificActivityModel.givensAnglesFeaturesPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=numFreeParamsPath, testingLoss=None, trainingHolder=model.specificActivityModel.numFreeParams, testingHolder=None)
                self.accelerator.print("Activity loss values:", activityTrainingLoss.nanmean().item(), activityTestingLoss.nanmean().item())

                # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
                emotionTestingLoss = self.calculateEmotionLoss(emotionProfile, allLabels, allTestingLabelMask)  # Dim: numEmotions
                emotionTrainingLoss = self.calculateEmotionLoss(emotionProfile, allLabels, allTrainingLabelMask)  # Dim: numEmotions

                # Get the encoder information.
                givensAnglesPath, normalizationFactorsPath, givensAnglesFeaturesPath, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams(submodelString="EmotionModel")
                activationParamsPath, moduleNames = model.getActivationParamsFullPassPath(submodelString="EmotionModel")
                numFreeParamsPath, _, _ = model.getFreeParamsFullPassPath(submodelString="EmotionModel")
                # givensAnglesFeaturesPath: numModuleLayers, numFeatures=5, numValues*
                # normalizationFactorsPath: numModuleLayers, numSignals, numParams=1
                # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                # givensAnglesPath: numModuleLayers, numSignals, numParams
                # activationParamsPath: numActivations, numParams=3

                # Store the signal encoder loss information.
                self.storeLossInformation(trainingLoss=emotionTrainingLoss, testingLoss=emotionTestingLoss, trainingHolder=model.specificEmotionModel.trainingLosses_signalReconstruction, testingHolder=model.specificEmotionModel.testingLosses_signalReconstruction)
                self.storeLossInformation(trainingLoss=activationParamsPath, testingLoss=None, trainingHolder=model.specificEmotionModel.activationParamsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=normalizationFactorsPath, testingLoss=None, trainingHolder=model.specificEmotionModel.normalizationFactorsPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=givensAnglesFeaturesPath, testingLoss=None, trainingHolder=model.specificEmotionModel.givensAnglesFeaturesPath, testingHolder=None)
                self.storeLossInformation(trainingLoss=numFreeParamsPath, testingLoss=None, trainingHolder=model.specificEmotionModel.numFreeParams, testingHolder=None)
                self.accelerator.print("Emotion loss values:", emotionTrainingLoss.nanmean().item(), emotionTestingLoss.nanmean().item())

    # --------------------------- Helper Methods --------------------------- #

    def storeLossInformation(self, trainingLoss, testingLoss, trainingHolder, testingHolder):
        # Gather the losses from each device (for distributed training). NOTE: don't use gather as it doesn't remove duplicates.
        gatheredTrainingLoss, gatheredTestingLoss = self.accelerator.gather_for_metrics((trainingLoss, testingLoss))

        # Store the loss information.
        if isinstance(gatheredTrainingLoss, torch.Tensor):
            trainingHolder.append(gatheredTrainingLoss.detach().cpu().numpy().astype(np.float16))
            if testingLoss is not None: testingHolder.append(gatheredTestingLoss.detach().cpu().numpy().astype(np.float16))
        else:
            trainingHolder.append(gatheredTrainingLoss)
            if testingLoss is not None: testingHolder.append(gatheredTestingLoss)

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

            emotionClassWeight = torch.ones_like(emotionClassWeight)  # TODO
            allEmotionClassWeights.append(emotionClassWeight)

        # Assign the class weights for the activities.
        activityLabels = self.dataInterface.getActivityLabels(allLabels, validLabelsMask, self.activityLabelInd)
        activityClassWeights = self.assignClassWeights(activityLabels, numActivities)
        activityClassWeights = torch.ones_like(activityClassWeights)  # TODO

        # Set the loss function for the activity and emotion classes.
        self.setEmotionActivityLossFunctions(activityClassWeights=activityClassWeights, emotionClassWeights=allEmotionClassWeights)

        return allEmotionClassWeights, activityClassWeights

    @staticmethod
    def assignClassWeights(class_labels, num_classes):
        # Remove the invalid labels.
        validLabels = ~torch.isnan(class_labels)
        validClassLabels = class_labels[validLabels]

        # Assert that the class labels are valid.
        if len(validClassLabels) == 0: return torch.nan
        assert validClassLabels.max() < num_classes, "The class labels must be less than the number of classes."
        assert 0 <= validClassLabels.min(), "The class labels must be non-negative."
        assert class_labels.ndim == 1, "The class labels must be 1D."

        # Initialize the class counts.
        lowerBoundLabels = validClassLabels.floor()
        upperBoundLabels = validClassLabels.ceil()

        # Get the upper and lower-class weights.
        upperBoundWeight = validClassLabels - lowerBoundLabels  # [0, 1]
        lowerBoundWeight = 1 - upperBoundWeight  # [0, 1]

        # Remove invalid lower and upper bounds.
        upperBoundLabels[num_classes - 1 <= upperBoundLabels] = num_classes - 1
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
