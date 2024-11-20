import time

import torch

from .emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionPipelineHelpers import emotionPipelineHelpers


class emotionPipeline(emotionPipelineHelpers):

    def __init__(self, accelerator, datasetName, allEmotionClasses, numSubjects, userInputParams,
                 emotionNames, activityNames, featureNames, submodel, numExperiments):
        # General parameters.
        super().__init__(accelerator=accelerator, datasetName=datasetName, allEmotionClasses=allEmotionClasses, numSubjects=numSubjects, userInputParams=userInputParams,
                         emotionNames=emotionNames, activityNames=activityNames, featureNames=featureNames, submodel=submodel, numExperiments=numExperiments)
        # Finish setting up the model.
        self.compileOptimizer(submodel)  # Initialize the optimizer (for back propagation)

    def trainModel(self, dataLoader, submodel, inferenceTraining, profileTraining, specificTraining, trainSharedLayers, stepScheduler, numEpochs):
        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        # allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata, reconstructionIndex = self.prepareInformation(dataLoader)
        # allEmotionClassWeights, activityClassWeights = self.organizeLossInfo.getClassWeights(allLabels, allTrainingMasks, allTestingMasks, self.numActivities)
        self.setupTraining(submodel, inferenceTraining=inferenceTraining, profileTraining=profileTraining, specificTraining=specificTraining, trainSharedLayers=trainSharedLayers)
        if self.model.debugging: self.accelerator.print(f"\nTraining {self.datasetName} model")
        datasetSpecificTraining = profileTraining and (specificTraining or trainSharedLayers)
        onlyProfileTraining = profileTraining and not (specificTraining or trainSharedLayers)

        # For each training epoch.
        for epoch in range(numEpochs):
            numPointsAnalyzed = 0

            # For each data batch in the epoch.
            for batchDataInd, batchData in enumerate(dataLoader):
                with (self.accelerator.accumulate(self.model)):  # Accumulate the gradients.
                    with self.accelerator.autocast():  # Enable mixed precision auto-casting
                        # Extract the data, labels, and testing/training indices.
                        if not inferenceTraining: batchSignalInfo, batchSignalLabels, batchTrainingLabelMask, batchTestingLabelMask, batchTrainingSignalMask, batchTestingSignalMask = self.extractBatchInformation(batchData)
                        else: batchSignalInfo = batchData; batchTrainingLabelMask, batchTestingLabelMask, batchTrainingSignalMask, batchTestingSignalMask = None, None, None, None
                        if onlyProfileTraining: batchTrainingLabelMask, batchTestingLabelMask, batchTrainingSignalMask, batchTestingSignalMask = None, None, None, None

                        # We can skip this batch, and backpropagation if necessary.
                        if batchSignalInfo.size(0) == 0: self.backpropogateModel(); continue
                        numPointsAnalyzed += batchSignalInfo.size(0)

                        # Set the training parameters.
                        signalBatchData, batchSignalIdentifiers, metaBatchInfo = emotionDataInterface.separateData(batchSignalInfo)
                        # signalBatchData[:, :, :, 0] = timepoints: [further away from survey (300) -> closest to survey (0)]
                        # signalBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                        # batchSignalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
                        # metaBatchInfo dimension: batchSize, numMetadata

                        if not inferenceTraining:
                            with torch.no_grad():
                                # Augment the signals to train an arbitrary sequence length and order.
                                augmentedBatchData = self.dataAugmentation.changeNumSignals(signalBatchData, dropoutPercent=0.1)
                                augmentedBatchData = self.dataAugmentation.signalDropout(augmentedBatchData, dropoutPercent=0.1)
                                # augmentedBatchData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                        else: augmentedBatchData = signalBatchData

                        # ------------ Forward pass through the model  ------------- #

                        t11 = time.time()
                        # Perform the forward pass through the model.
                        validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = self.model.forward(submodel, augmentedBatchData, batchSignalIdentifiers, metaBatchInfo, device=self.accelerator.device, inferenceTraining=inferenceTraining)
                        # reconstructedSignalData dimension: batchSize, numSignals, maxSequenceLength
                        # basicEmotionProfile: batchSize, numBasicEmotions, encodedDimension
                        # validDataMask dimension: batchSize, numSignals, maxSequenceLength
                        # physiologicalProfile dimension: batchSize, encodedDimension
                        # resampledSignalData dimension: batchSize, encodedDimension
                        # activityProfile: batchSize, numActivities, encodedDimension
                        # emotionProfile: batchSize, numEmotions, encodedDimension
                        t22 = time.time()

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(reconstructedSignalData, variableName="reconstructed signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(physiologicalProfile, variableName="physiological profile", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(resampledSignalData, variableName="resampled signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(basicEmotionProfile, variableName="basic emotion profile", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(activityProfile, variableName="activity profile", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(emotionProfile, variableName="emotion profile", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(validDataMask, variableName="valid data mask", assertGradient=False)

                        # Calculate the error in signal compression (signal encoding loss).
                        trainingSignalReconstructedLosses = self.organizeLossInfo.calculateSignalEncodingLoss(augmentedBatchData, reconstructedSignalData, validDataMask, batchTrainingSignalMask)
                        if trainingSignalReconstructedLosses is None: self.accelerator.print("Not useful loss"); continue
                        finalTrainingLoss = trainingSignalReconstructedLosses.nanmean()

                        # Initialize basic core loss value.
                        if self.model.debugging: self.accelerator.print("Final loss:", finalTrainingLoss.item())

                        # ------------------- Update the Model  -------------------- #

                        t1 = time.time()
                        if datasetSpecificTraining:
                            # Calculate the error in signal compression (signal encoding loss).
                            testingSignalReconstructedLosses = self.organizeLossInfo.calculateSignalEncodingLoss(augmentedBatchData, reconstructedSignalData, validDataMask, batchTestingSignalMask)
                            finalTestingLoss = testingSignalReconstructedLosses.nanmean() if testingSignalReconstructedLosses is not None else None
                            assert batchTestingSignalMask is not None, "The testing signal mask must be defined."
                            self.accelerator.backward(finalTestingLoss, retain_graph=True)

                            # For each layer in the model.
                            for layerName, layerParams in self.model.named_parameters():
                                # Remove the gradients from the shared and specific layers.
                                if layerParams.grad is not None and 'profileModel' not in layerName: layerParams.grad.zero_()

                        # Update the model parameters.
                        self.accelerator.backward(finalTrainingLoss)  # Calculate the gradients.
                        self.backpropogateModel()  # Backpropagation.
                        if self.model.debugging: t2 = time.time(); self.accelerator.print(f"{'Shared' if trainSharedLayers else '\tSpecific'} layer training {self.datasetName} {numPointsAnalyzed}: {t22 - t11} {t2 - t1}\n")

        # Update the learning rate.
        if stepScheduler: self.scheduler.step()

        # Prepare the model/data for evaluation.
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Turn off training flags.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

        return emotionProfile

    def backpropogateModel(self):
        if self.accelerator.sync_gradients:
            # Clip the gradients to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Backpropagation the gradient.
            self.optimizer.step()  # Adjust the weights.
            self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            if self.model.debugging: self.accelerator.print(f"Backprop with LR: {self.scheduler.get_last_lr()}")
