# General
import concurrent.futures
import random
import time
import torch
from matplotlib import pyplot as plt

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

    def trainEpoch(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders):
        # Set random order to loop through the models.
        self.unifyAllModelWeights(allMetaModels, allModels)
        modelIndices = list(range(len(allMetaModels)))
        random.shuffle(modelIndices)

        # For each training model.
        for modelInd in modelIndices:
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            trainSharedLayers = modelInd < len(allMetaModels)  # Train the shared layers.

            # Train the updated model.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], modelWeights=self.sharedModelWeights, layerInfo=self.unifiedLayerData)
            modelPipeline.trainModel(dataLoader, submodel, profileTraining=False, specificTraining=True, trainSharedLayers=trainSharedLayers, stepScheduler=False, numEpochs=1)   # Full model training.
            self.accelerator.wait_for_everyone()

            # Unify all the model weights and retrain the specific models.
            modelPipeline.modelHelpers.roundModelWeights(modelPipeline.model, decimals=8)
            self.unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)
        self.datasetSpecificTraining(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, profileOnlyTraining=False)

    def datasetSpecificTraining(self, submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, profileOnlyTraining=False):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)
        if allMetaModels[0].model.numSpecificEncoderLayers == 0:
            print("No specific training needed.")
            profileOnlyTraining = True

        # For each meta-training model.
        for modelInd in range(len(allMetaModels) + len(allModels)):
            dataLoader = allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders) else allDataLoaders[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(allMetaModels) else allModels[modelInd - len(allMetaModels)]  # Same pipeline instance in training loop.
            if modelPipeline.datasetName.lower() == 'empatch': numEpochs = 2
            elif modelPipeline.datasetName.lower() == 'wesad': numEpochs = 4
            else: numEpochs = 1

            # Train the updated model.
            if not profileOnlyTraining: modelPipeline.trainModel(dataLoader, submodel, profileTraining=False, specificTraining=True, trainSharedLayers=False, stepScheduler=False, numEpochs=numEpochs)  # Signal-specific training.

            # Health profile training.
            numProfileShots = modelPipeline.resetPhysiologicalProfile(submodel)
            modelPipeline.trainModel(dataLoader, submodel, profileTraining=True, specificTraining=False, trainSharedLayers=False, stepScheduler=True, numEpochs=numProfileShots + 1)  # Profile training.
            self.accelerator.wait_for_everyone()

    def boundAngularWeights(self, allMetaModels, allModels, applyMinThresholding, doubleThresholding):
        # Unify all the model weights.
        self.unifyAllModelWeights(allMetaModels, allModels)

        # For each meta-training model.
        for modelPipeline in allMetaModels + allModels:
            modelPipeline.model.cullAngles(applyMinThresholding=applyMinThresholding, doubleThresholding=doubleThresholding)

        # Unify all the model weights and retrain the specific models.
        self.unifiedLayerData = self.modelMigration.copyModelWeights(allMetaModels[0], self.sharedModelWeights)
        self.unifyAllModelWeights(allMetaModels, allModels)

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
        """ Parallelized version of the function for efficient plotting. """
        self.unifyAllModelWeights(allMetaModels, allModels)  # Unify model weights before plotting.
        hpcFlag = 'HPC' in modelConstants.userInputParams['deviceListed']  # Whether we are using the HPC.
        numModels = len(allMetaModels) + len(allModels)
        t1 = time.time()
        plt.close('all')

        def process_model(modelInd):
            """ Function to process and plot a model in parallel. """
            # Select appropriate data loader and model pipeline
            lossDataLoader = (allMetadataLoaders[modelInd] if modelInd < len(allMetadataLoaders)
                              else allDataLoaders[modelInd - len(allMetaModels)])
            modelPipeline = (allMetaModels[modelInd] if modelInd < len(allMetaModels)
                             else allModels[modelInd - len(allMetaModels)])

            with torch.no_grad():  # Disable gradient computation for efficiency
                numEpochs = modelPipeline.getTrainingEpoch(submodel)
                modelPipeline.modelVisualization.plotAllTrainingEvents(
                    submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs, showMinimumPlots=showMinimumPlots if hpcFlag else False
                )

        def process_model_comparison():
            """ Function to plot the model comparison separately in parallel. """
            with torch.no_grad():
                allMetaModels[0].modelVisualization.plotDatasetComparison(
                    submodel, allMetaModels + allModels, trainingDate, showMinimumPlots=showMinimumPlots
                )

        # Use ThreadPoolExecutor to run everything in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_model, modelInd) for modelInd in range(numModels)]
            concurrent.futures.wait(futures)
        process_model_comparison()

        t2 = time.time()
        self.accelerator.print("Total plotting time:", t2 - t1)

    def plotModelState_Old(self, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingDate, showMinimumPlots):
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
