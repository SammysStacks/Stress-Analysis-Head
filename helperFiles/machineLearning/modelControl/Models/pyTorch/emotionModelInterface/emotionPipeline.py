import random
import time

from .emotionModel.emotionModelHelpers.modelConstants import modelConstants
from .emotionPipelineHelpers import emotionPipelineHelpers


class emotionPipeline(emotionPipelineHelpers):

    def __init__(self, accelerator, datasetName, allEmotionClasses, numSubjects, userInputParams,
                 emotionNames, activityNames, featureNames, submodel, numExperiments, reconstructionIndex):
        # General parameters.
        super().__init__(accelerator=accelerator, datasetName=datasetName, allEmotionClasses=allEmotionClasses, numSubjects=numSubjects, userInputParams=userInputParams,
                         emotionNames=emotionNames, activityNames=activityNames, featureNames=featureNames, submodel=submodel, numExperiments=numExperiments)
        # General parameters.
        self.reconstructionIndex = reconstructionIndex  # The index of the signal to reconstruct.
        self.augmentData = True

        # Finish setting up the model.
        self.compileOptimizer(submodel)  # Initialize the optimizer (for back propagation)

    def trainModel(self, dataLoader, submodel, trainSharedLayers, inferenceTraining, profileTraining, numEpochs):
        """
        Stored items in the dataLoader.dataset:
            allData: The standardized testing and training data → Dim: numExperiments, numSignals, totalLength, numChannels
            allLabels: Integer labels representing class indices. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
            allTestingMasks: Boolean flags representing if the label is a testing label. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
            allTrainingMasks: Boolean flags representing if the label is a training label. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
                Note: totalLength = finalDistributionLength + 1 + demographicLength (The extra +1 is for the subject index)
                Note: the last dimension in allLabels is for human activity recognition.
        """
        self.accelerator.print(f"\nTraining {self.datasetName} model", flush=True)

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        # allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata, reconstructionIndex = self.prepareInformation(dataLoader)
        # allEmotionClassWeights, activityClassWeights = self.organizeLossInfo.getClassWeights(allLabels, allTrainingMasks, allTestingMasks, self.numActivities)
        self.setupTraining(submodel, trainSharedLayers=trainSharedLayers, inferenceTraining=inferenceTraining, profileTraining=profileTraining)

        # For each training epoch.
        for epoch in range(numEpochs):
            numPointsAnalyzed = 0

            # For each data batch in the epoch.
            for batchDataInd, batchData in enumerate(dataLoader):
                with self.accelerator.accumulate(self.model):  # Accumulate gradients.
                    # Extract the data, labels, and testing/training indices.
                    if not inferenceTraining: batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = self.extractBatchInformation(batchData)
                    else: batchSignalInfo = batchData; batchTrainingMask, batchTestingMask = None, None

                    # We can skip this batch, and backpropagation if necessary.
                    if batchSignalInfo.size(0) == 0: self.backpropogateModel(submodel); continue
                    numPointsAnalyzed += batchSignalInfo.size(0)

                    # Separate the data into signal and metadata information.
                    currentTrainingMask = batchTrainingMask if not profileTraining or not inferenceTraining else None
                    signalBatchData, batchSignalIdentifiers, metaBatchInfo = self.dataInterface.separateData(batchSignalInfo)
                    # signalBatchData[:, :, :, 0] = timepoints: [further away from survey (300) -> closest to survey (0)]
                    # signalBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    # batchSignalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
                    # metaBatchInfo dimension: batchSize, numMetadata

                    # Adjust the data precision.
                    signalBatchData = signalBatchData.double()

                    # For every new batch.
                    if not inferenceTraining and self.accelerator.sync_gradients: self.augmentData = random.uniform(a=0, b=1) < 0.5

                    if not inferenceTraining and self.augmentData:
                        # Augment the signals to train an arbitrary sequence length and order.
                        augmentedBatchData = self.dataAugmentation.changeNumSignals(signalBatchData, dropoutPercent=0.25)
                        augmentedBatchData = self.dataAugmentation.signalDropout(augmentedBatchData, dropoutPercent=0.25)
                        # augmentedBatchData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    else: augmentedBatchData = signalBatchData

                    # ------------ Forward pass through the model  ------------- #

                    # Perform the forward pass through the model.
                    validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = self.model.forward(submodel, augmentedBatchData, batchSignalIdentifiers, metaBatchInfo, device=self.accelerator.device, inferenceTraining=inferenceTraining)
                    # reconstructedSignalData dimension: batchSize, numSignals, maxSequenceLength
                    # basicEmotionProfile: batchSize, numBasicEmotions, encodedDimension
                    # validDataMask dimension: batchSize, numSignals, maxSequenceLength
                    # physiologicalProfile dimension: batchSize, encodedDimension
                    # resampledSignalData dimension: batchSize, encodedDimension
                    # activityProfile: batchSize, numSignals, encodedDimension
                    # emotionProfile: batchSize, numEmotions, encodedDimension

                    # Assert that nothing is wrong with the predictions.
                    self.modelHelpers.assertVariableIntegrity(reconstructedSignalData, variableName="reconstructed signal data", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(physiologicalProfile, variableName="physiological profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(resampledSignalData, variableName="resampled signal data", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(basicEmotionProfile, variableName="basic emotion profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(activityProfile, variableName="activity profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(emotionProfile, variableName="emotion profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(validDataMask, variableName="valid data mask", assertGradient=False)

                    # Calculate the error in signal compression (signal encoding loss).
                    physiologicalSmoothLoss, resampledSmoothLoss = self.organizeLossInfo.calculateSmoothLoss(physiologicalProfile, resampledSignalData, validDataMask, currentTrainingMask, self.reconstructionIndex)
                    signalReconstructedLoss = self.organizeLossInfo.calculateSignalEncodingLoss(augmentedBatchData, reconstructedSignalData, validDataMask, currentTrainingMask, self.reconstructionIndex)
                    if signalReconstructedLoss is None: self.accelerator.print("Not useful loss"); continue

                    # Initialize basic core loss value.
                    finalLoss = signalReconstructedLoss + 0.05*(physiologicalSmoothLoss + resampledSmoothLoss)

                    # Update the user.
                    self.accelerator.print("Final-Recon-Phys-Resamp", finalLoss.item(), signalReconstructedLoss.item(), physiologicalSmoothLoss.item(), resampledSmoothLoss.item(), flush=True)

                    # ------------------- Update the Model  -------------------- #

                    # Prevent exploding loss values.
                    while 2 < finalLoss.item(): finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    self.backpropogateModel(submodel)  # Backpropagation.
                    t2 = time.time(); self.accelerator.print(f"{'Shared' if trainSharedLayers else '\tSpecific'} layer training {self.datasetName} {numPointsAnalyzed}: {t2 - t1}\n")

        # Prepare the model/data for evaluation.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

    def backpropogateModel(self, submodel):
        # Clip the gradients if they are too large.
        if self.accelerator.sync_gradients:
            if submodel == modelConstants.signalEncoderModel: self.modelHelpers.scaleGradients(self.model)
            # self.accelerator.clip_grad_norm_(self.model.parameters(), 20)  # Apply gradient clipping: Small: <1; Medium: 5-10; Large: >20

            # Backpropagation the gradient.
            self.optimizer.step()  # Adjust the weights.
            self.scheduler.step()  # Update the learning rate.
            self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            self.accelerator.print(f"Backprop with LR: {self.scheduler.get_last_lr()}", flush=True)
        
    def extractBatchInformation(self, batchData):
        # Extract the data, labels, and testing/training indices.
        batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = batchData
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.accelerator.device), batchTestingMask.to(self.accelerator.device)
        batchSignalInfo, batchSignalLabels = batchSignalInfo.to(self.accelerator.device), batchSignalLabels.to(self.accelerator.device)
        
        return batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask
