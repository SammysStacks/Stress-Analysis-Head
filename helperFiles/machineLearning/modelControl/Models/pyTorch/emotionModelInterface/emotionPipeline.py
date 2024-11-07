import random
import time

from .emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
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

    def trainModel(self, dataLoader, submodel, inferenceTraining, profileTraining, specificTraining, trainSharedLayers, numEpochs):
        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        # allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata, reconstructionIndex = self.prepareInformation(dataLoader)
        # allEmotionClassWeights, activityClassWeights = self.organizeLossInfo.getClassWeights(allLabels, allTrainingMasks, allTestingMasks, self.numActivities)
        self.setupTraining(submodel, inferenceTraining=inferenceTraining, profileTraining=profileTraining, specificTraining=specificTraining, trainSharedLayers=trainSharedLayers)
        self.accelerator.print(f"\nTraining {self.datasetName} model", flush=True)

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
                    if batchSignalInfo.size(0) == 0: self.backpropogateModel(); continue
                    numPointsAnalyzed += batchSignalInfo.size(0)

                    # Set the training parameters.
                    if profileTraining and not specificTraining and not trainSharedLayers: currentTrainingMask = None
                    elif inferenceTraining: currentTrainingMask = None
                    else: currentTrainingMask = batchTrainingMask

                    # Unpack the batch data information.
                    signalBatchData, batchSignalIdentifiers, metaBatchInfo = emotionDataInterface.separateData(batchSignalInfo)
                    # signalBatchData[:, :, :, 0] = timepoints: [further away from survey (300) -> closest to survey (0)]
                    # signalBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    # batchSignalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
                    # metaBatchInfo dimension: batchSize, numMetadata

                    # For every new batch.
                    if not inferenceTraining and self.accelerator.sync_gradients: self.augmentData = random.uniform(a=0, b=1) < 0.25
                    signalBatchData = signalBatchData.double()  # Convert the data to double precision.

                    if not inferenceTraining and self.augmentData:
                        # Augment the signals to train an arbitrary sequence length and order.
                        augmentedBatchData = self.dataAugmentation.changeNumSignals(signalBatchData, dropoutPercent=0.2)
                        augmentedBatchData = self.dataAugmentation.signalDropout(augmentedBatchData, dropoutPercent=0.2)
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
                    # activityProfile: batchSize, numActivities, encodedDimension
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
                    finalLoss = signalReconstructedLoss # + 0.1*(physiologicalSmoothLoss + resampledSmoothLoss)
                    self.accelerator.print("Final-Recon-Phys-Resamp", finalLoss.item(), signalReconstructedLoss.item(), physiologicalSmoothLoss.item(), resampledSmoothLoss.item(), flush=True)

                    # ------------------- Update the Model  -------------------- #

                    # Prevent exploding loss values.
                    while 2 < finalLoss.item(): finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    self.backpropogateModel()  # Backpropagation.

                    t2 = time.time(); self.accelerator.print(f"{'Shared' if trainSharedLayers else '\tSpecific'} layer training {self.datasetName} {numPointsAnalyzed}: {t2 - t1}\n")

        # Prepare the model/data for evaluation.
        self.accelerator.wait_for_everyone()  # Wait before continuing.
        return emotionProfile

    def backpropogateModel(self):
        # Clip the gradients if they are too large.
        if self.accelerator.sync_gradients:

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.norm().item(), param.grad.max().item(), param.grad.min().item())

            # Backpropagation the gradient.
            self.optimizer.step()  # Adjust the weights.
            self.scheduler.step()  # Update the learning rate.

            self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            self.accelerator.print(f"Backprop with LR: {self.scheduler.get_last_lr()}", flush=True)
