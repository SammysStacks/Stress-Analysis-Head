# General
import torch
import time

# Helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration


class trainingProtocolHelpers:

    def __init__(self, submodel, accelerator, sharedModelWeights):
        # General parameters.
        self.submodelsSaving = modelParameters.getSubmodelsSaving(submodel)  # The submodels to save.
        self.sharedModelWeights = sharedModelWeights
        self.accelerator = accelerator
        self.unifiedLayerData = None

        # Helper classes.
        self.modelMigration = modelMigration(accelerator)
        self.modelHelpers = modelHelpers()

    def trainEpoch(self, submodel, allMetadataLoaders, allMetaModels, allModels):
        # For each meta-training model.
        for modelInd in range(len(allMetadataLoaders)):
            dataLoader = allMetadataLoaders[modelInd]
            modelPipeline = allMetaModels[modelInd]

            # Train the updated model.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], sharedModelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
            modelPipeline.trainModel(dataLoader, submodel, numEpochs=1)
            self.accelerator.wait_for_everyone()

            # Store the new model weights.
            self.unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

    def calculateLossInformation(self, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, fastPass):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaLossDataHolders)):
            lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

            # Calculate and store all the training and testing losses of the untrained model.
            with torch.no_grad(): modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader, fastPass)
        t2 = time.time(); self.accelerator.print("Total loss calculation time:", t2 - t1)

    def plotModelState(self, epoch, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, trainingDate, fastPass=True):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaLossDataHolders)):
            lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

            with torch.no_grad():
                numEpochs = modelPipeline.getTrainingEpoch(submodel) or epoch
                modelPipeline.modelVisualization.plotAllTrainingEvents(submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs, fastPass)
        allMetaModels[0].modelVisualization.plotDatasetComparison(submodel, allMetaModels + allModels, trainingDate, fastPass)
        t2 = time.time()
        self.accelerator.print("Total plotting time:", t2 - t1)

    def saveModelState(self, epoch, allMetaModels, allModels, submodel, modelName, allDatasetNames, trainingDate):
        # Prepare to save the model.
        numEpochs = allMetaModels[-1].getTrainingEpoch(submodel) or epoch
        self.unifyAllModelWeights(allMetaModels, allModels)
        allPipelines = allMetaModels + allModels

        # Save the current version of the model.
        self.modelMigration.saveModels(modelPipelines=allPipelines, modelName=modelName, datasetNames=allDatasetNames, sharedModelWeights=self.sharedModelWeights, submodelsSaving=self.submodelsSaving,
                                       submodel=submodel, trainingDate=trainingDate, numEpochs=numEpochs, metaTraining=True, saveModelAttributes=True, storeOptimizer=False)

    def unifyAllModelWeights(self, allMetaModels=None, allModels=None):
        if self.unifiedLayerData is None: self.unifiedLayerData = self.modelMigration.copyModelWeights(allMetaModels[0], self.sharedModelWeights)

        # Unify all the model weights.
        if allMetaModels: self.modelMigration.unifyModelWeights(allModels=allMetaModels, sharedModelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
        if allModels: self.modelMigration.unifyModelWeights(allModels=allModels, sharedModelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)

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
