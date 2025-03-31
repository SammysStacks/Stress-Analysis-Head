# General
import os
import sys
import accelerate
import torch

# Set specific environmental parameters.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import plotting methods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData
from helperMethods.signalEncoderPlots import signalEncoderPlots


if __name__ == "__main__":
    # General model parameters.
    trainingDate = "2025-03-11"  # The current date we are training the model. Unique identifier of this training set.
    holdDatasetOut = True  # Whether to hold out the validation dataset.
    testSplitRatio = 0.1  # The percentage of testing points.

    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # Define the model information.
    modelFolder = os.path.dirname(os.path.abspath(__file__)) + "/../../../helperFiles/machineLearning/_finalModels/signalEncoderModel/"
    submodel = modelConstants.signalEncoderModel

    # --------------------------- Model Parameters -------------------------- #

    # Initialize the model information classes.
    modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator)  # Initialize the model compiler.
    trainingProtocols = trainingProtocolHelpers(submodel=submodel, accelerator=accelerator)  # Initialize the training protocols.
    modelParameters = modelParameters(accelerator)  # Initialize the model parameters class.
    modelMigration = modelMigration(accelerator)  # Initialize the model migration class.

    # Compile the final modules.
    datasetNames, metaDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    allModels, allDataLoaders, allMetaModels, allMetadataLoaders, _ = modelCompiler.compileModelsFull(metaDatasetNames, submodel, testSplitRatio, datasetNames, loadSubmodelDate)
    allDataLoaders.append(allMetadataLoaders.pop(0))  # Do not metatrain with wesad data.
    datasetNames.append(metaDatasetNames.pop(0))  # Do not metatrain with wesad data.
    allModels.append(allMetaModels.pop(0))  # Do not metatrain with wesad data.

    # Compile all the datasets together.
    if holdDatasetOut: allDataLoaders, datasetNames, allModels = [], [], []
    allDatasetNames = metaDatasetNames + datasetNames

    # --------------------------- Figure Plotting -------------------------- #

    # Initialize plotting classes.
    saveFolder = os.path.dirname(os.path.abspath(__file__)) + "/finalFigures/"
    signalEncoderPlots = signalEncoderPlots(modelName, datasetNames, sharedModelWeights, savingBaseFolder=saveFolder, accelerator=accelerate.Accelerator(cpu=True))

