# General
import torch
import time

# Helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration


class trainingProtocolHelpers:

    def __init__(self, submodel, accelerator):
        # General parameters.
        self.submodelsSaving = modelParameters.getSubmodelsSaving(submodel)  # The submodels to save.
        self.specificModelWeights = modelConstants.specificModelWeights  # The dataset-specific model weights.
        self.sharedModelWeights = modelConstants.sharedModelWeights  # The shared model weights.
        self.minEpochs_modelAdjustment = 1  # The minimum number of epochs before adjusting the model architecture.
        self.accelerator = accelerator
        self.unifiedLayerData = None
        self.numTrailingLosses = 2

        # Helper classes.
        self.modelMigration = modelMigration(accelerator)
        self.modelHelpers = modelHelpers()

    def adjustModelArchitecture(self, allMetaModels, allModels):
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]

            # Get the model losses.
            modelLosses = modelPipeline.model.specificSignalEncoderModel.trainingLosses_signalReconstruction
            if len(modelLosses) < 2*self.numTrailingLosses: return None

            # Check if we should add a new layer.
            lossPercent = sum(modelLosses[-self.numTrailingLosses:]) / sum(modelLosses[-self.numTrailingLosses*2:-self.numTrailingLosses])
            if not (0.95 < lossPercent < 1): return None
            if modelLosses[-1] < 0.1: return None

        # Add a new layer to the model.
        self.addModelLayer(allMetaModels, allModels)

    def addModelLayer(self, allMetaModels, allModels):
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            # Train the updated model.
            modelPipeline.model.addNewLayer()

        self.unifiedLayerData = None
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

    @staticmethod
    def inferenceTraining(modelPipeline, inputData, numEpochs):
        # Prepare the model for inference training.
        inputData = torch.as_tensor(inputData, dtype=torch.float64)
        numExperiments, numSignals, sequenceLength, numChannels = inputData.size()
        modelPipeline.model.inferenceModel.resetInferenceModel(numExperiments)
        experimentalInds = torch.arange(0, numExperiments, dtype=torch.int64)
        dataLoader = zip(experimentalInds, inputData)

        # Train the inference model.
        modelPipeline.trainModel(dataLoader, submodel=modelConstants.signalEncoderModel, inferenceTraining=True, trainSharedLayers=False, numEpochs=numEpochs)

    def trainEpoch(self, submodel, allMetadataLoaders, allMetaModels, allModels):
        # For each meta-training model.
        for modelInd in range(len(allMetadataLoaders)):
            dataLoader = allMetadataLoaders[modelInd]
            modelPipeline = allMetaModels[modelInd]

            # Train the updated model.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
            modelPipeline.trainModel(dataLoader, submodel, inferenceTraining=False, trainSharedLayers=True, numEpochs=1)
            self.accelerator.wait_for_everyone()

            # Train the specific layers.
            self.trainSpecificLayers(submodel, allMetadataLoaders, allMetaModels, allModels)

            # Store the new model weights.
            self.unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

    def trainSpecificLayers(self, submodel, allMetadataLoaders, allMetaModels, allModels):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

        # For each meta-training model.
        for modelInd in range(len(allMetadataLoaders)):
            dataLoader = allMetadataLoaders[modelInd]
            modelPipeline = allMetaModels[modelInd]

            # Train the updated model.
            modelPipeline.trainModel(dataLoader, submodel, inferenceTraining=False, trainSharedLayers=False, numEpochs=1)
            self.accelerator.wait_for_everyone()

    def calculateLossInformation(self, allMetaModels, allMetadataLoaders, allModels, allDataLoaders, submodel):
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify all the model weights.

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            lossDataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.

            # Calculate and store all the training and testing losses of the untrained model.
            with torch.no_grad(): modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader)
        t2 = time.time(); self.accelerator.print("Total loss calculation time:", t2 - t1)

    def plotModelState(self, allMetaModels, allMetadataLoaders, allModels, allDataLoaders, submodel, trainingDate):
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
