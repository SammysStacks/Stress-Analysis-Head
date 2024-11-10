# General
import random

import torch
import time

from torch.utils.data import dataloader

# Helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration


class trainingProtocolHelpers:

    def __init__(self, submodel, accelerator):
        # General parameters.
        self.submodelsSaving = modelParameters.getSubmodelsSaving(submodel)  # The submodels to save.
        self.sharedModelWeights = modelConstants.sharedModelWeights  # The shared model weights.
        self.minEpochs_modelAdjustment = 1  # The minimum number of epochs before adjusting the model architecture.
        self.accelerator = accelerator
        self.unifiedLayerData = None
        self.numTrailingLosses = 2

        # Helper classes.
        self.modelMigration = modelMigration(accelerator)
        self.modelHelpers = modelHelpers()

    @staticmethod
    def inferenceTraining(dataLoader, modelPipeline, submodel, inputData, encodedDimension, numEpochs):
        # Prepare the model for inference training.
        inputData = torch.as_tensor(inputData, dtype=torch.float64)
        numExperiments, numSignals, sequenceLength, numChannels = inputData.size()
        modelPipeline.model.inferenceModel.resetInferenceModel(numExperiments, encodedDimension)

        # Train the inference model.
        emotionProfile = modelPipeline.trainModel(dataLoader, submodel=submodel,  inferenceTraining=True, profileTraining=False, specificTraining=False, trainSharedLayers=False, numEpochs=numEpochs)
        return emotionProfile

    def trainEpoch(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders):
        # Set random order to loop through the models.
        self.unifyAllModelWeights(allMetaModels, allModels)
        modelIndices = list(range(len(allModels) + len(allMetaModels)))
        random.shuffle(modelIndices)

        # For each training model.
        for modelInd in modelIndices:
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            trainSharedLayers = modelInd < len(allMetaModels)  # Train the shared layers.

            # Train the updated model.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
            modelPipeline.trainModel(dataLoader, submodel, inferenceTraining=False, profileTraining=True, specificTraining=True, trainSharedLayers=trainSharedLayers, numEpochs=1)   # Full model training.
            self.accelerator.wait_for_everyone()

            # Store the new model weights.
            self.unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights and retrain the specific models.
        self.datasetSpecificTraining(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders)
        self.unifyAllModelWeights(allMetaModels, allModels)

    def datasetSpecificTraining(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            # Train the updated model.
            modelPipeline.trainModel(dataLoader, submodel, inferenceTraining=False, profileTraining=True, specificTraining=False, trainSharedLayers=False, numEpochs=2)  # Signal-specific training: training only.
            self.accelerator.wait_for_everyone()

    def calculateLossInformation(self, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            lossDataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            # Calculate and store all the training and testing losses of the untrained model.
            with torch.no_grad(): modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader)
        t2 = time.time(); self.accelerator.print("Total loss calculation time:", t2 - t1)

    def plotModelState(self, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingDate):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            lossDataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            with torch.no_grad():
                numEpochs = modelPipeline.getTrainingEpoch(submodel)
                modelPipeline.modelVisualization.plotAllTrainingEvents(submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs)
        allMetaModels[0].modelVisualization.plotDatasetComparison(submodel, allMetaModels + allModels, trainingDate)
        t2 = time.time()
        self.accelerator.print("Total plotting time:", t2 - t1)

    def saveModelState(self, epoch, allMetaModels, allModels, submodel, allDatasetNames, trainingDate):
        # Prepare to save the model.
        numEpochs = allMetaModels[-1].getTrainingEpoch(submodel) or epoch
        self.unifyAllModelWeights(allMetaModels, allModels)
        allPipelines = allMetaModels + allModels

        # Save the current version of the model.
        self.modelMigration.saveModels(modelPipelines=allPipelines, datasetNames=allDatasetNames, sharedModelWeights=self.sharedModelWeights, submodelsSaving=self.submodelsSaving,
                                       submodel=submodel, trainingDate=trainingDate, numEpochs=numEpochs, metaTraining=True, saveModelAttributes=True, storeOptimizer=False)

    def unifyAllModelWeights(self, allMetaModels=None, allModels=None):
        if self.unifiedLayerData is None: self.unifiedLayerData = self.modelMigration.copyModelWeights(allMetaModels[0], self.sharedModelWeights)

        # Unify all the model weights.
        if allMetaModels: self.modelMigration.unifyModelWeights(allModels=allMetaModels, modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
        if allModels: self.modelMigration.unifyModelWeights(allModels=allModels, modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)

    # DEPRECATED
    def constrainSpectralNorm(self, allMetaModels, allModels, unifiedLayerData, addingSN):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

        # For each meta-training model.
        for modelPipeline in allMetaModels:
            self.modelHelpers.hookSpectralNormalization(modelPipeline.model, n_power_iterations=5, addingSN=addingSN)

        # For each training model.
        for modelPipeline in allModels:
            self.modelHelpers.hookSpectralNormalization(modelPipeline.model, n_power_iterations=5, addingSN=addingSN)

        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

        return unifiedLayerData
