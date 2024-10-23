""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import os

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants

# Set specific environmental parameters.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# General
import accelerate
import argparse
import torch
import time

# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData  # Methods to organize model data.

# Configure cuDNN and PyTorch's global settings.
torch.backends.cudnn.deterministic = True  # If False: allow non-deterministic algorithms in cuDNN, which can enhance performance but reduce reproducibility.
torch.autograd.set_detect_anomaly(True)  # If True: detect NaN values in the output of autograd.
torch.set_default_dtype(torch.float64)  # Set the default data type to float32, which is typical for neural network computations.
torch.backends.cudnn.benchmark = False  # If True: Enable cuDNN's auto-tuner to find the most efficient algorithm for the current configuration, potentially improving performance if fixed input size.

if __name__ == "__main__":
    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # General model parameters.
    trainingDate = "2024-10-02 wavelet analysis"  # The current date we are training the model. Unique identifier of this training set.
    modelName = "emotionModel"  # The emotion model's unique identifier. Options: emotionModel
    testSplitRatio = 0.2  # The percentage of testing points.

    # Training flags.
    storeLoss = True  # If you want to record any loss values.

    # ----------------------- Parse Model Parameters ----------------------- #

    # Create the parser
    parser = argparse.ArgumentParser(description='Specify model parameters.')

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default=modelConstants.signalEncoderModel, help='The component of the model we are training. Options: signalEncoderModel, emotionModel')
    parser.add_argument('--optimizerType', type=str, default='AdamW', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc.')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are running the platform on')
    parser.add_argument('--reversibleLearningProtocol', type=str, default='rFC', help='The learning rate for the optimizer.')
    parser.add_argument('--irreversibleLearningProtocol', type=str, default='rFC', help='The learning rate for the optimizer.')

    # Add arguments for the neural operator.
    parser.add_argument('--operatorType', type=str, default='wavelet', help='The type of operator to use for the neural operator: wavelet')
    parser.add_argument('--waveletType', type=str, default='bior3.7', help='The wavelet type for the wavelet transform: bior3.7, db3, dmey, etc')

    # Add arguments for the signal encoder prediction
    parser.add_argument('--encodedDimension', type=int, default=512, help='The dimension of the encoded signal.')

    # Add arguments for the emotion prediction.
    parser.add_argument('--numBasicEmotions', type=int, default=8, help='The number of basic emotions (basis states of emotions).')

    # Parse the arguments.
    userInputParams = vars(parser.parse_args())
    submodel = userInputParams['submodel']
    print("Arguments:", userInputParams)

    # Compile additional input parameters.
    userInputParams = modelParameters.getNeuralParameters(userInputParams)

    # --------------------------- Setup Training --------------------------- #

    # Initialize the model information classes.
    trainingProtocols = trainingProtocolHelpers(submodel=submodel, accelerator=accelerator)  # Initialize the training protocols.
    modelCompiler = compileModelData(submodel, userInputParams, useTherapyData=False, accelerator=accelerator)
    modelParameters = modelParameters(userInputParams, accelerator)  # Initialize the model parameters class.
    modelMigration = modelMigration(accelerator)  # Initialize the model migration class.

    # Specify training parameters
    numEpochs, numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo()  # The number of epochs to plot and save the model.
    datasetNames, metaDatasetNames, allDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    trainingDate = modelCompiler.embedInformation(submodel, trainingDate)  # Embed training information into the name.

    # Compile the final modules.
    allModels, allDataLoaders, allMetaModels, allMetadataLoaders, _ = modelCompiler.compileModelsFull(metaDatasetNames, modelName, submodel, testSplitRatio, datasetNames)

    # Store the initial loss information.
    if storeLoss: trainingProtocols.calculateLossInformation(allMetaModels, allMetadataLoaders, allModels, allDataLoaders, submodel)

    # -------------------------- Meta-model Training ------------------------- #

    # For each training epoch
    for epoch in range(1, numEpochs + 1):
        print(f"\nEpoch: {epoch}", flush=True)
        startEpochTime = time.time()

        # Get the saving information.
        saveFullModel, plotSteps = modelParameters.getSavingInformation(epoch, numEpoch_toSaveFull, numEpoch_toPlot)

        # Train the model for a single epoch.
        trainingProtocols.trainEpoch(submodel, allMetadataLoaders, allMetaModels, allModels)

        # Store the initial loss information and plot.
        if storeLoss: trainingProtocols.calculateLossInformation(allMetaModels, allMetadataLoaders, allModels, allDataLoaders, submodel)
        if plotSteps: trainingProtocols.plotModelState(allMetaModels, allMetadataLoaders, allModels, allDataLoaders, submodel, trainingDate)

        # Save the model sometimes (only on the main device).
        # if saveFullModel and accelerator.is_local_main_process:
        #     trainingProtocols.saveModelState(epoch, allMetaModels, allModels, submodel, modelName, allDatasetNames, trainingDate)

        # Adjust the model architecture if needed.
        # trainingProtocols.adjustModelArchitecture(allMetaModels, allModels)

        # Finalize the epoch parameters.
        accelerator.wait_for_everyone()  # Wait before continuing.
        endEpochTime = time.time()

        print("Total epoch time:", endEpochTime - startEpochTime)

    # -------------------------- Empatch Training ------------------------- #

    # # Unify all the fixed weights in the models
    # unifiedLayerData = modelMigration.copyModelWeights(modelPipeline, sharedModelWeights=modelConstants.sharedModelWeights)
    # modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights=modelConstants.sharedModelWeights, unifiedLayerData)
    # modelMigration.unifyModelWeights(allModels, sharedModelWeights=modelConstants.sharedModelWeights, unifiedLayerData)
    #
    # # SHAP analysis on the metalearning models.
    # featureAnalysis = _featureImportance.featureImportance(modelCompiler.saveTrainingData)
    #
    # # For each metatraining model.
    # for modelInd in metaModelIndices:
    #     dataLoader = allMetadataLoaders[modelInd]
    #     modelPipeline = allMetaModels[modelInd]
    #     # Place model in eval mode.
    #     modelPipeline.model.eval()
    #
    #     # Extract all the data.
    #     allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
    #
    #     # Stop gradient tracking.
    #     with torch.no_grad():
    #         # Convert time-series to features.
    #         compressedData, encodedData, transformedData, signalFeatures, subjectInds = modelPipeline.model.compileSignalFeatures(
    #             allData, fullDataPass=False)
    #
    #     # Reshape the signal features to match the SHAP format
    #     reshapedSignalFeatures = signalFeatures.view(len(signalFeatures), -1)
    #     # reshapedSignalFeatures.view(signalFeatures.shape)
    #
    #     featureAnalysis.shapAnalysis(modelPipeline.model, reshapedSignalFeatures, allLabels,
    #                                  featureNames=modelPipeline.featureNames, modelType="", shapSubfolder="")
