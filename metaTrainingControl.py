""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """
import math
import os
import sys
# Set specific environmental parameters.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stderr = sys.stdout  # Redirect stderr to stdout

# General
import accelerate
import argparse
import torch
import time

# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData

# Configure cuDNN and PyTorch's global settings.
torch.backends.cudnn.deterministic = False  # If True: ensures that the model will be reproducible.
torch.autograd.set_detect_anomaly(False)  # If True: detect NaN values in the output of autograd. Will be slower.
torch.backends.cudnn.benchmark = False  # Enable cuDNN's auto-tuner to find the most efficient algorithm. Keep true for fixed input sizes.

if __name__ == "__main__":
    # Read in any user input parameters.
    parser = argparse.ArgumentParser(description='Specify model parameters')

    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # General model parameters.
    trainingDate = "2025-04-01 shared-2"  # The current date we are training the model. Unique identifier of this training set.
    plotAllEpochs = False  # Whether to plot all data every epoch (plotting once every numEpoch_toPlot regardless).
    validationRun = False  # Whether to train new datasets from the old model.
    testSplitRatio = 0.1  # The percentage of testing points.

    # Model loading information.
    loadSubmodelDate = "2025-03-30"  # The submodel we are loading: None, "2025-03-31"

    # ----------------------- Architecture Parameters ----------------------- #

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default=modelConstants.signalEncoderModel, help='The component of the model we are training. Options: signalEncoderModel, emotionModel')
    parser.add_argument('--optimizerType', type=str, default='NAdam', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc')
    parser.add_argument('--learningProtocol', type=str, default='reversibleLieLayer', help='The learning protocol for the model: reversibleLieLayer')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are using: cpu, cuda')

    # Add arguments for the health profile.
    parser.add_argument('--initialProfileAmp', type=float, default=0.01, help='The limits for profile initialization. Should be near zero')
    parser.add_argument('--encodedDimension', type=int, default=256, help='The dimension of the health profile and all signals.')
    parser.add_argument('--numProfileShots', type=int, default=16, help='The epochs for profile training: [16, 32]')
    
    # Add arguments for the neural operator.
    parser.add_argument('--waveletType', type=str, default='bior3.1', help='The wavelet type for the wavelet transform: bior3.1, bior3.3, bior2.2, bior3.5')
    parser.add_argument('--operatorType', type=str, default='wavelet', help='The type of operator to use for the neural operator: wavelet')

    # Add arguments for the signal encoder architecture.
    parser.add_argument('--numSpecificEncoderLayers', type=int, default=1, help='The number of layers in the model: [1, 2]')
    parser.add_argument('--numSharedEncoderLayers', type=int, default=7, help='The number of layers in the model: [2, 10]')

    # Add arguments for observational learning.
    parser.add_argument('--maxAngularThreshold', type=float, default=45, help='The larger rotational threshold in (degrees)')
    parser.add_argument('--minAngularThreshold', type=float, default=2, help='The smaller rotational threshold in (degrees)')

    # dd arguments for the emotion and activity architecture.
    parser.add_argument('--numBasicEmotions', type=int, default=6, help='The number of basic emotions (basis states of emotions)')
    parser.add_argument('--numActivityModelLayers', type=int, default=4, help='The number of layers in the activity model')
    parser.add_argument('--activityLearningRate', type=float, default=0.1, help='The learning rate of the activity model')
    parser.add_argument('--numEmotionModelLayers', type=int, default=4, help='The number of layers in the emotion model')
    parser.add_argument('--emotionLearningRate', type=float, default=0.01, help='The learning rate of the emotion model')
    parser.add_argument('--numActivityChannels', type=int, default=4, help='The number of activity  channels')

    # ----------------------- Training Parameters ----------------------- #

    # Signal encoder learning rates.
    parser.add_argument('--profileLR', type=float, default=0.01, help='The learning rate of the profile')
    parser.add_argument('--physGenLR', type=float, default=1e-3, help='The learning rate of the profile generation (CNNs)')
    parser.add_argument('--reversibleLR', type=float, default=0.067, help='The learning rate of the Lie manifold angles (degrees)')

    # Add arguments for the emotion and activity architecture.
    parser.add_argument('--momentum_decay', type=float, default=0.001, help='Momentum decay for the optimizer')
    parser.add_argument('--beta1', type=float, default=0.7, help='Beta1 for the optimizer: 0.5 -> 0.99')  # 0.6, 0.7, 0.8
    parser.add_argument('--beta2', type=float, default=0.8, help='Beta2 for the optimizer: 0.9 -> 0.999')  # 0.8, 0.9

    # ----------------------- Compile Parameters ----------------------- #

    # Parse the arguments.
    userInputParams = vars(parser.parse_args())
    userInputParams['minWaveletDim'] = userInputParams['encodedDimension'] // (2**3)
    userInputParams['minThresholdStep'] = userInputParams['reversibleLR']  # Keep as degrees
    userInputParams['reversibleLR'] = userInputParams['reversibleLR'] * math.pi / 180  # Keep as radians
    userInputParams['profileDimension'] = userInputParams['encodedDimension'] // 2

    # Compile additional input parameters.
    userInputParams = modelParameters.getNeuralParameters(userInputParams)
    modelConstants.updateModelParams(userInputParams)
    submodel = userInputParams['submodel']
    print("Arguments:", userInputParams)

    # Initialize the model information classes.
    trainingProtocols = trainingProtocolHelpers(submodel=submodel, accelerator=accelerator)  # Initialize the training protocols.
    modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator)  # Initialize the model compiler.
    modelMigrationClass = modelMigration(accelerator)  # Initialize the model migration class.

    # Specify training parameters
    numEpochs, numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo(validationRun)  # The number of epochs to plot and save the model.
    trainingModelName = modelParameters.embedInformation(submodel, trainingDate)  # Embed training information into the name.
    datasetNames, metaDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    print("modelName", trainingModelName, "\n")

    # Compile the final modules.
    allModels, allDataLoaders, allMetaModels, allMetadataLoaders, _ = modelCompiler.compileModelsFull(metaDatasetNames, submodel, testSplitRatio, datasetNames, loadSubmodelDate)
    allDataLoaders.append(allMetadataLoaders.pop(0))  # Do not metatrain with wesad data.
    datasetNames.append(metaDatasetNames.pop(0))  # Do not metatrain with wesad data.
    allModels.append(allMetaModels.pop(0))  # Do not metatrain with wesad data.

    # Do not train on the meta-datasets.
    if not validationRun: allDataLoaders, datasetNames, allModels = [], [], []
    else: allMetadataLoaders, metaDatasetNames, allMetaModels = [], [], []
    allDatasetNames = metaDatasetNames + datasetNames

    # -------------------------- Meta-model Training ------------------------- #

    # Plot the initial model state.
    if modelConstants.useInitialLoss: trainingProtocols.calculateLossInformation(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel)  # Calculate the initial loss.
    if plotAllEpochs: trainingProtocols.plotModelState(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingModelName, showMinimumPlots=False)
    # if not validationRun: trainingProtocols.datasetSpecificTraining(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch=0, onlyProfileTraining=True)

    # For each training epoch
    for epoch in range(1, numEpochs + 1):
        print(f"\nEpoch: {epoch}")
        startEpochTime = time.time()

        # Get the saving information.
        saveFullModel, showAllPlots = modelParameters.getEpochParameters(epoch, numEpoch_toSaveFull, numEpoch_toPlot, plotAllEpochs)

        # Train the model for a single epoch.
        if not validationRun: trainingProtocols.trainEpoch(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch)
        else: trainingProtocols.validationTraining(submodel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch)

        # Store the initial loss information and plot.
        trainingProtocols.calculateLossInformation(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel)
        trainingProtocols.plotModelState(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingModelName, showMinimumPlots=not showAllPlots)

        # Save the model sometimes (only on the main device).
        if saveFullModel and accelerator.is_local_main_process:
            trainingProtocols.saveModelState(epoch, allMetaModels, allModels, submodel, allDatasetNames, trainingDate)

        # Finalize the epoch parameters.
        endEpochTime = time.time(); print("Total epoch time:", endEpochTime - startEpochTime)

    # -------------------------- SHAP Analysis ------------------------- #

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
