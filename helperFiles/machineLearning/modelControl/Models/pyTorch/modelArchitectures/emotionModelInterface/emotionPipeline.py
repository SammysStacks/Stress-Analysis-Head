# General
import random
import time

from .emotionModel.emotionModelHelpers.modelConstants import modelConstants
# Import files for machine learning
from .emotionPipelineHelpers import emotionPipelineHelpers


class emotionPipeline(emotionPipelineHelpers):

    def __init__(self, accelerator, modelID, datasetName, modelName, allEmotionClasses, maxNumSignals, numSubjects, userInputParams,
                 emotionNames, activityNames, featureNames, submodel, useFinalParams, debuggingResults=False):
        # General parameters.
        super().__init__(accelerator, modelID, datasetName, modelName, allEmotionClasses, maxNumSignals, numSubjects, userInputParams,
                         emotionNames, activityNames, featureNames, submodel, useFinalParams, debuggingResults)
        # General parameters.
        self.maxBatchSignals = maxNumSignals
        self.calculateFullLoss = False
        self.addingNoiseFlag = False

        # Finish setting up the model.
        self.modelHelpers.l2Normalization(self.model, maxNorm=20, checkOnly=True)
        self.modelHelpers.switchActivationLayers(self.model, switchState=True)
        self.compileOptimizer(submodel)  # Initialize the optimizer (for back propagation)

    def trainModel(self, dataLoader, submodel, numEpochs=500, constrainedTraining=False):
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
        allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allMetadata, reconstructionIndex = self.prepareInformation(dataLoader)
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
                    numPointsAnalyzed += batchSignalInfo.size(0)

                    # Interface for non-emotion modeling, where only the signal data is used (no labels).
                    if submodel in [modelConstants.signalEncoderModel, modelConstants.autoencoderModel]:
                        batchTrainingMask, batchSignalLabels, batchSignalInfo = self.dataInterface.getReconstructionData(batchTrainingMask, batchSignalLabels, batchSignalInfo, reconstructionIndex)
                        
                        # If there is no training data.
                        if batchSignalInfo.size(0) == 0:
                            # We can skip this batch, and backpropagate the model if necessary.
                            if self.accelerator.sync_gradients: self.backpropogateModel()
                            continue

                    # Separate the data into signal and metadata information.
                    signalBatchData, batchSignalIdentifiers, metaBatchInfo = self.dataInterface.separateData(batchSignalInfo)
                    # signalBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    # batchSignalIdentifiers dimension: batchSize, numSignals, numSignalIdentifiers
                    # metaBatchInfo dimension: batchSize, numMetadata

                    # Randomly choose to add noise to the model.
                    if self.accelerator.sync_gradients and not constrainedTraining:
                        self.addingNoiseFlag = submodel == modelConstants.emotionPredictionModel and random.random() < 0.5
                        self.calculateFullLoss = random.random() < 0.5

                    # Randomly choose to add noise to the model.
                    if self.addingNoiseFlag and not constrainedTraining:
                        # Augment the data to add some noise to the model.
                        addingNoiseSTD, addingNoiseRange = self.modelParameters.getAugmentationDeviation(submodel)
                        augmentedBatchData = self.dataAugmentation.addNoise(signalBatchData.clone(), trainingFlag=True, noiseSTD=addingNoiseSTD)
                        # augmentedBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                    else:
                        addingNoiseSTD, addingNoiseRange = 0, (0, 1)
                        augmentedBatchData = signalBatchData.clone()

                    # ------------ Forward pass through the model  ------------- #

                    # Train the signal encoder
                    if submodel == modelConstants.signalEncoderModel:
                        # Randomly choose to use an inflated number of signals.
                        if self.accelerator.sync_gradients:
                            self.maxBatchSignals = random.choices(population=[modelConstants.maxNumSignals, signalBatchData.shape[1]], weights=[0.6, 0.4], k=1)[0]

                        # Augment the signals to train an arbitrary sequence length and order.
                        augmentedBatchData = self.dataAugmentation.changeNumSignals(signalBatchData, minNumSignals=model.numEncodedSignals, maxNumSignals=self.maxBatchSignals, alteredDim=1)
                        batchStartTimeIndices = self.dataAugmentation.getNewStartTimeIndices(signalData=augmentedBatchData, minTimeWindow=modelConstants.timeWindows[0], maxTimeWindow=modelConstants.timeWindows[-1])
                        # augmentedBatchData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
                        # batchStartTimeIndices dimension: batchSize, numSignals
                        print("Input size:", augmentedBatchData.size())

                        # Perform the forward pass through the model.
                        encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss = model.signalEncoding(augmentedBatchData, batchStartTimeIndices, batchSignalIdentifiers, metaBatchInfo, decodeSignals=True, calculateLoss=self.calculateFullLoss, trainingFlag=True)
                        # decodedPredictedIndexProbabilities dimension: batchSize, numSignals, maxNumEncodedSignals
                        # predictedIndexProbabilities dimension: batchSize, numSignals, maxNumEncodedSignals
                        # encodedData dimension: batchSize, numEncodedSignals, finalDistributionLength
                        # reconstructedData dimension: batchSize, numSignals, finalDistributionLength
                        # signalEncodingLayerLoss dimension: batchSize

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(predictedIndexProbabilities, variableName="signal encoder index probabilities", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(signalEncodingLayerLoss, variableName="signal encoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedData, variableName="reconstructed signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(augmentedBatchData, variableName="initial signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(encodedData, variableName="encoded data", assertGradient=False)

                        # Calculate the error in signal compression (signal encoding loss).
                        signalReconstructedLoss, encodedSignalMeanLoss, encodedSignalMinMaxLoss, positionalEncodingTrainingLoss, decodedPositionalEncodingLoss, signalEncodingTrainingLayerLoss \
                            = self.organizeLossInfo.calculateSignalEncodingLoss(augmentedBatchData, encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss, batchTrainingMask, reconstructionIndex)
                        if signalReconstructedLoss.item() == 0: self.accelerator.print("Not useful\n\n\n\n\n\n"); continue

                        # Initialize basic core loss value.
                        compressionFactor = augmentedBatchData.size(1) / encodedData.size(1)  # Increase the learning rate for larger compressions.
                        finalLoss = compressionFactor * signalReconstructedLoss

                        # Compile the loss into one value
                        if 0.3 < encodedSignalMinMaxLoss:
                            finalLoss = finalLoss + 0.1*encodedSignalMinMaxLoss
                        if 0.1 < signalEncodingTrainingLayerLoss:
                            finalLoss = finalLoss + 0.25*signalEncodingTrainingLayerLoss
                        if 0.1 < encodedSignalMeanLoss:
                            finalLoss = finalLoss + encodedSignalMeanLoss
                        if signalReconstructedLoss < 0.1 < decodedPositionalEncodingLoss:
                            finalLoss = finalLoss + 0.25*decodedPositionalEncodingLoss
                        # Account for the current training state when calculating the loss.
                        finalLoss = finalLoss + 0.25*positionalEncodingTrainingLoss

                        # Update the user.
                        self.accelerator.print("Final-Recon-Mean-MinMax-PE-PEDec-Layer", finalLoss.item(), signalReconstructedLoss.item(), encodedSignalMeanLoss.item(), encodedSignalMinMaxLoss.item(), positionalEncodingTrainingLoss.item(), decodedPositionalEncodingLoss.item(), signalEncodingTrainingLayerLoss.item(), "\n")

                    # Train the autoencoder
                    elif submodel == modelConstants.autoencoderModel:
                        # Augment the time series length to train an arbitrary sequence length.
                        initialSignalData, augmentedBatchData = self.dataAugmentation.changeSignalLength(modelConstants.timeWindows[0], (signalBatchData, augmentedBatchData))
                        print("Input size:", augmentedBatchData.size())

                        # Perform the forward pass through the model.
                        encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss = \
                            model.compressData(augmentedBatchData, initialSignalData, reconstructSignals=True, calculateLoss=True, compileVariables=False, compileLosses=False, fullReconstruction=True, trainingFlag=True)
                        # denoisedDoubleReconstructedData dimension: batchSize, numSignals, finalDistributionLength
                        # reconstructedEncodedData dimension: batchSize, numEncodedSignals, finalDistributionLength
                        # compressedData dimension: batchSize, numEncodedSignals, compressedLength
                        # autoencoderLayerLoss dimension: batchSize

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(denoisedDoubleReconstructedData, variableName="denoised double reconstructed data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedEncodedData, variableName="reconstructed encoded data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(autoencoderLayerLoss, variableName="autoencoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(compressedData, variableName="compressed data", assertGradient=False)

                        # Calculate the error in signal reconstruction (autoencoder loss).
                        encodedReconstructedLoss, compressedMeanLoss, compressedMinMaxLoss, autoencoderTrainingLayerLoss = \
                            self.organizeLossInfo.calculateAutoencoderLoss(encodedData, compressedData, reconstructedEncodedData, autoencoderLayerLoss, batchTrainingMask, reconstructionIndex)
                        # Calculate the error in signal reconstruction (encoding loss).
                        signalReconstructedLoss = self.organizeLossInfo.signalEncodingLoss(initialSignalData, denoisedDoubleReconstructedData).mean(dim=2).mean(dim=1).mean()

                        # Initialize basic core loss value.
                        compressionFactorSE = augmentedBatchData.size(1) / self.model.numEncodedSignals
                        compressionFactor = augmentedBatchData.size(2) / self.model.compressedLength
                        finalLoss = encodedReconstructedLoss

                        # Compile the loss into one value
                        if 0.1 < compressedMinMaxLoss:
                            finalLoss = finalLoss + 0.1 * compressedMinMaxLoss
                        if 0.01 < autoencoderTrainingLayerLoss:
                            finalLoss = finalLoss + 0.5 * autoencoderTrainingLayerLoss
                        if 0.1 < compressedMeanLoss:
                            finalLoss = finalLoss + 0.1 * compressedMeanLoss
                        finalLoss = compressionFactor * (finalLoss + compressionFactorSE * signalReconstructedLoss)

                        # Update the user.
                        self.accelerator.print(finalLoss.item(), encodedReconstructedLoss.item(), compressedMeanLoss.item(), compressedMinMaxLoss.item(), autoencoderTrainingLayerLoss.item(), signalReconstructedLoss.item(), "\n")

                    elif submodel == modelConstants.emotionPredictionModel:
                        # Perform the forward pass through the model.
                        _, _, _, compressedData, _, _, _, mappedSignalData, reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
                            = model.emotionPrediction(augmentedBatchData, signalBatchData, metaBatchInfo, remapSignals=True, compileVariables=False, trainingFlag=True)
                        # eachBasicEmotionDistribution dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength
                        # finalEmotionDistributions dimension: self.numEmotions, batchSize, self.emotionLength
                        # activityDistribution dimension: batchSize, self.numActivities
                        # featureData dimension: batchSize, self.numCommonFeatures

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(featureData, "feature data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(mappedSignalData, "mapped signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(activityDistribution, "activity distribution", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(finalEmotionDistributions, "final emotion distributions", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(eachBasicEmotionDistribution, "basic emotion distributions", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedCompressedData, "reconstructed compressed data", assertGradient=False)

                        # Calculate the error in emotion and activity prediction models.
                        manifoldReconstructedLoss, manifoldMeanLoss, manifoldMinMaxLoss = self.organizeLossInfo.calculateSignalMappingLoss(
                            encodedData, manifoldData, transformedManifoldData, reconstructedEncodedData, batchTrainingMask, reconstructionIndex)
                        emotionLoss, emotionOrthogonalityLoss, modelSpecificWeights = self.organizeLossInfo.calculateEmotionsLoss(activityDistribution, batchSignalLabels, batchTrainingMask, activityClassWeights)
                        activityLoss = self.organizeLossInfo.calculateActivityLoss(activityDistribution, batchSignalLabels, batchTrainingMask, activityClassWeights)

                        # Compile the loss into one value
                        manifoldLoss = 0.8 * manifoldReconstructedLoss + 0.1 * manifoldMeanLoss + 0.1 * manifoldMinMaxLoss
                        finalLoss = emotionLoss * 0.45 + emotionOrthogonalityLoss * 0.05 + modelSpecificWeights * 0.05 + activityLoss * 0.4 + manifoldLoss * 0.05
                    else:
                        raise Exception()

                    # ------------------- Update the Model  -------------------- #

                    # Prevent too high losses from randomizing weights.
                    while 10 < finalLoss: finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    self.backpropogateModel()  # Backpropagate the gradient.
                    t2 = time.time(); self.accelerator.print(f"Backprop {self.datasetName} {numPointsAnalyzed}:", t2 - t1)
            # Finalize all the parameters.
            self.scheduler.step()  # Update the learning rate.

        # Prepare the model/data for evaluation.
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Set all models into evaluation mode.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

    def backpropogateModel(self):
        # Clip the gradients if they are too large.
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 20)  # Apply gradient clipping: Small: <1; Medium: 5-10; Large: >20

        # Backpropagation the gradient.
        self.optimizer.step()  # Adjust the weights.
        self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
        self.accelerator.print("LR:", self.scheduler.get_last_lr())
        
    def extractBatchInformation(self, batchData):
        # Extract the data, labels, and testing/training indices.
        batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = batchData
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.accelerator.device), batchTestingMask.to(self.accelerator.device)
        batchSignalInfo, batchSignalLabels = batchSignalInfo.to(self.accelerator.device), batchSignalLabels.to(self.accelerator.device)
        
        return batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask
