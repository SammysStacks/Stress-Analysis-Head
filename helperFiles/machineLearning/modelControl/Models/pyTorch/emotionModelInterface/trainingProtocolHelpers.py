# General
import random
import time

import torch

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
        self.accelerator = accelerator
        self.unifiedLayerData = None

        # Helper classes.
        self.modelMigration = modelMigration(accelerator)
        self.modelHelpers = modelHelpers

    def trainEpoch(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch):
        # Set random order to loop through the models.
        self.unifyAllModelWeights(allMetaModels, allModels)
        modelIndices = list(range(len(allMetaModels)))
        random.shuffle(modelIndices)

        # For each training model.
        for modelInd in modelIndices:
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            trainSharedLayers = modelInd < len(allMetaModels)  # Train the shared layers.

            # Copy over the shared layers.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)

            # Train the updated model.
            modelPipeline.trainModel(dataLoader, submodel, profileTraining=False, specificTraining=True, trainSharedLayers=trainSharedLayers, stepScheduler=False, numEpochs=1)   # Full model training.
            self.accelerator.wait_for_everyone()

            # Unify all the model weights and retrain the specific models.
            self.unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)
        self.datasetSpecificTraining(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch, profileOnlyTraining=False)

    def datasetSpecificTraining(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch, profileOnlyTraining=False):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)
        if allMetaModels[0].model.numSpecificEncoderLayers == 0:
            print("No specific training needed.")
            profileOnlyTraining = True

        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            if modelPipeline.datasetName.lower() == 'empatch': numEpochs = 2  # [2, 3]
            elif modelPipeline.datasetName.lower() == 'wesad': numEpochs = 5  # 6
            else: numEpochs = 1

            # Train the updated model.
            if not profileOnlyTraining:
                modelPipeline.model.cullAngles(epoch=epoch)
                modelPipeline.trainModel(dataLoader, submodel, profileTraining=False, specificTraining=True, trainSharedLayers=False, stepScheduler=False, numEpochs=numEpochs)  # Signal-specific training.
                # modelPipeline.model.cullAngles(epoch=epoch)

            # Health profile training.
            numProfileShots = modelPipeline.resetPhysiologicalProfile(submodel)
            modelPipeline.trainModel(dataLoader, submodel, profileTraining=True, specificTraining=False, trainSharedLayers=False, stepScheduler=True, numEpochs=numProfileShots + 1)  # Profile training.
            self.accelerator.wait_for_everyone()
            if not profileOnlyTraining: modelPipeline.model.cullAngles(epoch=epoch)

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

    def plotModelState(self, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingDate, showMinimumPlots):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            lossDataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            with torch.no_grad():
                numEpochs = modelPipeline.getTrainingEpoch(submodel)
                modelPipeline.modelVisualization.plotAllTrainingEvents(submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs, showMinimumPlots=showMinimumPlots)
        with torch.no_grad(): allMetaModels[0].modelVisualization.plotDatasetComparison(submodel, allMetaModels + allModels, trainingDate, showMinimumPlots=showMinimumPlots)
        t2 = time.time()
        self.accelerator.print("Total plotting time:", t2 - t1)

    def saveModelState(self, epoch, allMetaModels, allModels, submodel, allDatasetNames, trainingDate):
        # Prepare to save the model.
        numEpochs = allMetaModels[-1].getTrainingEpoch(submodel) or epoch
        self.unifyAllModelWeights(allMetaModels, allModels)
        allPipelines = allMetaModels + allModels

        # Save the current version of the model.
        self.modelMigration.saveModels(modelPipelines=allPipelines, datasetNames=allDatasetNames, sharedModelWeights=self.sharedModelWeights, submodelsSaving=self.submodelsSaving,
                                       submodel=submodel, trainingDate=trainingDate, numEpochs=numEpochs, metaTraining=True, saveModelAttributes=True)

    def unifyAllModelWeights(self, allMetaModels=None, allModels=None):
        if self.unifiedLayerData is None: self.unifiedLayerData = self.modelMigration.copyModelWeights(allMetaModels[0], self.sharedModelWeights)

        # Unify all the model weights.
        if allMetaModels: self.modelMigration.unifyModelWeights(allModels=allMetaModels, modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
        if allModels: self.modelMigration.unifyModelWeights(allModels=allModels, modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
