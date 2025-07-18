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
    trainingDate = "2025-07-18 LR-3"  # The current date we are training the model. Unique identifier of this training set.
    unifyModelWeights = True  # Whether to unify the model weights across all models.
    plotAllEpochs = False  # Whether to plot all data every epoch (plotting once every numEpoch_toPlot regardless).
    validationRun = False  # Whether to train new datasets from the old model.
    testSplitRatio = 0.125  # The test split ratio for the emotion model is higher to allow more examples per class.

    # Model loading information.
    loadSubmodelDate = "2025-04-15---"  # The submodel we are loading: None, "2025-04-18"

    # ----------------------- Architecture Parameters ----------------------- #

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default=modelConstants.signalEncoderModel, help='The component of the model we are training. Options: signalEncoderModel, emotionModel')
    parser.add_argument('--optimizerType', type=str, default='NAdam', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc')
    parser.add_argument('--learningProtocol', type=str, default='reversibleLieLayer', help='The learning protocol for the model: reversibleLieLayer')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are using: cpu, cuda')

    # Add arguments for the health profile.
    parser.add_argument('--initialProfileAmp', type=float, default=1e-3, help='The limits for profile initialization. Should be near zero')
    parser.add_argument('--encodedDimension', type=int, default=512, help='The dimension of the health profile and all signals.')
    parser.add_argument('--numProfileShots', type=int, default=3, help='The epochs for profile training: [16, 32]')
    
    # Add arguments for the neural operator.
    parser.add_argument('--waveletType', type=str, default='bior3.1', help='The wavelet type for the wavelet transform: bior3.1, bior3.3, bior2.2, bior3.5')
    parser.add_argument('--operatorType', type=str, default='wavelet', help='The type of operator to use for the neural operator: wavelet')
    parser.add_argument('--numIgnoredSharedHF', type=int, default=0, help='The number of ignored high frequency components: [0, 1, 2]')

    # Add arguments for the signal encoder architecture.
    parser.add_argument('--numSpecificEncoderLayers', type=int, default=1, help='The number of layers in the model: [1, 2]')
    parser.add_argument('--numSharedEncoderLayers', type=int, default=7, help='The number of layers in the model: [2, 10]')

    # Add arguments for observational learning.
    parser.add_argument('--maxAngularThreshold', type=float, default=90, help='The larger rotational threshold in (degrees)')
    parser.add_argument('--minAngularThreshold', type=float, default=2, help='The smaller rotational threshold in (degrees)')

    # dd arguments for the emotion and activity architecture.
    parser.add_argument('--numBasicEmotions', type=int, default=4, help='The number of basic emotions (basis states of emotions)')
    parser.add_argument('--numActivityModelLayers', type=int, default=4, help='The number of layers in the activity model')
    parser.add_argument('--numEmotionModelLayers', type=int, default=4, help='The number of layers in the emotion model')

    # ----------------------- Training Parameters ----------------------- #

    # Signal encoder learning rates.
    parser.add_argument('--profileLR', type=float, default=0.01, help='The learning rate of the profile')
    parser.add_argument('--physGenLR', type=float, default=4e-4, help='The learning rate of the profile generation (CNNs)')
    parser.add_argument('--reversibleLR', type=float, default=0.05, help='The learning rate of the Lie manifold angles (degrees)')

    # Add arguments for the emotion and activity architecture.
    parser.add_argument('--momentum_decay', type=float, default=0.001, help='Momentum decay for the optimizer')
    parser.add_argument('--beta1', type=float, default=0.7, help='Beta1 for the optimizer: 0.5 -> 0.99')  # 0.6, 0.7, 0.8
    parser.add_argument('--beta2', type=float, default=0.8, help='Beta2 for the optimizer: 0.9 -> 0.999')  # 0.8, 0.9

    # ----------------------- Compile Parameters ----------------------- #

    # Parse the arguments.
    userInputParams = vars(parser.parse_args())
    userInputParams['minWaveletDim'] = max(32, userInputParams['encodedDimension'] // (2**4))
    userInputParams['minThresholdStep'] = userInputParams['reversibleLR']  # Keep as degrees
    userInputParams['reversibleLR'] = userInputParams['reversibleLR'] * math.pi / 180  # Keep as radians
    userInputParams['profileDimension'] = userInputParams['encodedDimension'] // 4  # The dimension of the profile.
    userInputParams['unifyModelWeights'] = unifyModelWeights

    # Compie additional input parameters.
    userInputParams = modelParameters.getNeuralParameters(userInputParams)
    modelConstants.updateModelParams(userInputParams)
    submodel = userInputParams['submodel']
    print("Arguments:", userInputParams)

    # Initialize the model information classes.
    modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator, validationRun=validationRun)  # Initialize the model compiler.
    trainingProtocols = trainingProtocolHelpers(submodel=submodel, accelerator=accelerator)  # Initialize the training protocols.
    modelMigrationClass = modelMigration(accelerator, validationRun)  # Initialize the model migration class.

    # Specify training parameters
    numEpochs, numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo(validationRun)  # The number of epochs to plot and save the model.
    trainingModelName = modelParameters.embedInformation(submodel, trainingDate, validationRun)  # Embed training information into the name.
    datasetNames, metaDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    print("modelName", trainingModelName, "\n")

    # Compile the final modules.
    allModels, allDataLoaders, allMetaModels, allMetadataLoaders, _ = modelCompiler.compileModelsFull(metaDatasetNames, submodel, testSplitRatio, datasetNames, loadSubmodelDate)
    allDataLoaders.append(allMetadataLoaders.pop(0))  # Do not metatrain with wesad data.
    datasetNames.append(metaDatasetNames.pop(0))  # Do not metatrain with wesad data.
    allModels.append(allMetaModels.pop(0))  # Do not metatrain with wesad data.
    allDatasetNames = metaDatasetNames + datasetNames

    # -------------------------- Meta-model Training ------------------------- #

    if submodel == modelConstants.emotionModel:
        # The emotion model needs to start with a health profile.
        for modelPipeline in (allMetaModels + allModels): modelPipeline.scheduler.scheduler.warmupFlag = False; modelPipeline.scheduler.scheduler.step()
        trainingProtocols.datasetSpecificTraining(modelConstants.signalEncoderModel, allMetadataLoaders, allMetaModels, allModels, allDataLoaders, epoch=0, onlyProfileTraining=True)
        trainingProtocols.plotModelState(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, modelConstants.signalEncoderModel, trainingModelName, showMinimumPlots=True)
        for modelPipeline in (allMetaModels + allModels): modelPipeline.scheduler.scheduler.warmupFlag = True; modelPipeline.scheduler.scheduler.step()

    # Plot the initial model state.
    trainingProtocols.calculateLossInformation(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel)  # Calculate the initial loss.
    if plotAllEpochs: trainingProtocols.plotModelState(allMetadataLoaders, allMetaModels, allModels, allDataLoaders, submodel, trainingModelName, showMinimumPlots=False)

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
        if saveFullModel and accelerator.is_local_main_process and submodel == modelConstants.signalEncoderModel:  # TODO
            trainingProtocols.saveModelState(epoch, allMetaModels, allModels, submodel, allDatasetNames, trainingDate)

        # Finalize the epoch parameters.
        endEpochTime = time.time(); print("Total epoch time:", endEpochTime - startEpochTime)
