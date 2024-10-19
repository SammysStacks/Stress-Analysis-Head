# General
import random
import time

from .emotionModel.emotionModelHelpers.modelConstants import modelConstants
# Import files for machine learning
from .emotionPipelineHelpers import emotionPipelineHelpers


class emotionPipeline(emotionPipelineHelpers):

    def __init__(self, accelerator, modelID, datasetName, modelName, allEmotionClasses, maxNumSignals, numSubjects, userInputParams,
                 emotionNames, activityNames, featureNames, submodel, numExperiments):
        # General parameters.
        super().__init__(accelerator, modelID, datasetName, modelName, allEmotionClasses, maxNumSignals, numSubjects, userInputParams,
                         emotionNames, activityNames, featureNames, submodel, numExperiments)
        # General parameters.
        self.augmentData = True

        # Finish setting up the model.
        self.modelHelpers.l2Normalization(self.model, maxNorm=20, checkOnly=True)
        self.modelHelpers.switchActivationLayers(self.model, switchState=True)
        self.compileOptimizer(submodel)  # Initialize the optimizer (for back propagation)

    def trainModel(self, dataLoader, submodel, numEpochs=500):
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
        allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata, reconstructionIndex = self.prepareInformation(dataLoader)
        allEmotionClassWeights, activityClassWeights = self.organizeLossInfo.getClassWeights(allLabels, allTrainingMasks, allTestingMasks, self.numActivities)

        # Prepare the model for training.
        model = self.getDistributedModel()
        self.setupTraining(submodel)

        # For each training epoch.
        for epoch in range(numEpochs):
            numPointsAnalyzed = 0

            # For each data batch in the epoch.
            for batchDataInd, batchData in enumerate(dataLoader):
                with self.accelerator.accumulate(model):  # Accumulate gradients.
                    # Extract the data, labels, and testing/training indices.
                    batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = self.extractBatchInformation(batchData)
                    batchSize, numSignals, maxSequenceLength, numChannels = batchSignalInfo.size()
                    numPointsAnalyzed += batchSize

                    # Get the reconstruction column for auto encoding and activity prediction.
                    if submodel == modelConstants.signalEncoderModel: batchTrainingMask, batchSignalLabels, batchSignalInfo = self.dataInterface.getReconstructionData(batchTrainingMask, batchSignalLabels, batchSignalInfo, reconstructionIndex)

                    # We can skip this batch, and backpropagation if necessary.
                    if batchSignalInfo.size(0) == 0: self.backpropogateModel(); continue

                    # Separate the data into signal and metadata information.
                    signalBatchData, batchSignalIdentifiers, metaBatchInfo = self.dataInterface.separateData(batchSignalInfo)
                    # signalBatchData[:, :, :, 0] = timepoints: [further away from survey (300) -> closest to survey (0)]
                    # signalBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    # batchSignalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
                    # metaBatchInfo dimension: batchSize, numMetadata

                    # Adjust the data precision.
                    signalBatchData = signalBatchData.round(decimals=6).double()
                    batchSignalIdentifiers = batchSignalIdentifiers.int()
                    metaBatchInfo = metaBatchInfo.double()

                    # For every new batch.
                    if self.accelerator.sync_gradients: self.augmentData = random.uniform(a=0, b=1) < 0.5

                    if self.augmentData:
                        # Augment the signals to train an arbitrary sequence length and order.
                        augmentedBatchData = self.dataAugmentation.changeNumSignals(signalBatchData, dropoutPercent=0.1)
                        augmentedBatchData = self.dataAugmentation.signalDropout(augmentedBatchData, dropoutPercent=0.1)
                        # augmentedBatchData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    else: augmentedBatchData = signalBatchData

                    # ------------ Forward pass through the model  ------------- #

                    # Get the physiological times.
                    physiologicalTimes = model.sharedSignalEncoderModel.pseudoEncodedTimes

                    # Perform the forward pass through the model.
                    missingDataMask, reconstructedSignalData, finalManifoldProjectionLoss, fourierData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, augmentedBatchData, batchSignalIdentifiers, metaBatchInfo, device=self.accelerator.device, trainingFlag=True)
                    # reconstructedSignalData dimension: batchSize, numSignals, encodedDimension
                    # physiologicalProfile dimension: batchSize, numSignals, encodedDimension
                    # fourierData dimension: batchSize, numEncodedSignals, fourierDimension
                    # missingDataMask dimension: batchSize, numSignals, maxSequenceLength
                    # basicEmotionProfile: batchSize, numBasicEmotions, encodedDimension
                    # activityProfile: batchSize, numSignals, encodedDimension
                    # emotionProfile: batchSize, numEmotions, encodedDimension
                    # finalManifoldProjectionLoss dimension: batchSize

                    # Assert that nothing is wrong with the predictions.
                    self.modelHelpers.assertVariableIntegrity(fourierData, variableName="up-sampled reconstructed signal data", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(finalManifoldProjectionLoss, variableName="manifold projected signals", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(physiologicalProfile, variableName="physiological profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(missingDataMask, variableName="missing data mask", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(activityProfile, variableName="activity profile", assertGradient=False)
                    self.modelHelpers.assertVariableIntegrity(emotionProfile, variableName="emotion profile", assertGradient=False)

                    # Calculate the error in signal compression (signal encoding loss).
                    signalReconstructedLoss, finalManifoldProjectionLoss = self.organizeLossInfo.calculateSignalEncodingLoss(augmentedBatchData, reconstructedSignalData, finalManifoldProjectionLoss, physiologicalTimes, missingDataMask, batchTrainingMask, reconstructionIndex)
                    if signalReconstructedLoss is None: self.accelerator.print("Not useful loss"); continue

                    # Initialize basic core loss value.
                    finalLoss = signalReconstructedLoss + finalManifoldProjectionLoss

                    # Update the user.
                    self.accelerator.print("Final-Recon", finalLoss.item(), signalReconstructedLoss.item(), finalManifoldProjectionLoss.item(), "\n")

                    # ------------------- Update the Model  -------------------- #

                    # Prevent too high losses from randomizing weights.
                    while 10 < finalLoss: finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    self.backpropogateModel()  # Backpropagation.
                    t2 = time.time(); self.accelerator.print(f"Backprop {self.datasetName} {numPointsAnalyzed}:", t2 - t1)

        # Prepare the model/data for evaluation.
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Set all models into evaluation mode.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

    def backpropogateModel(self):
        # Clip the gradients if they are too large.
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 20)  # Apply gradient clipping: Small: <1; Medium: 5-10; Large: >20
            self.accelerator.print("End batch", flush=True)

            # Backpropagation the gradient.
            self.optimizer.step()  # Adjust the weights.
            self.scheduler.step()  # Update the learning rate.
            self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            self.accelerator.print("LR:", self.scheduler.get_last_lr())
        
    def extractBatchInformation(self, batchData):
        # Extract the data, labels, and testing/training indices.
        batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = batchData
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.accelerator.device), batchTestingMask.to(self.accelerator.device)
        batchSignalInfo, batchSignalLabels = batchSignalInfo.to(self.accelerator.device), batchSignalLabels.to(self.accelerator.device)
        
        return batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask
