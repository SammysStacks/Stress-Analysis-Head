# General
import os
import sys
import accelerate

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants

# Set specific environmental parameters.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import plotting methods
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperMethods.signalEncoderPlots import signalEncoderPlots


if __name__ == "__main__":
    # General model parameters.
    trainingDate = "2025-03-11"  # The current date we are training the model. Unique identifier of this training set.
    holdDatasetOut = True  # Whether to hold out the validation dataset.
    testSplitRatio = 0.1  # The percentage of testing points.

    # --------------------------- User Input Parameters -------------------------- #

    submodel = modelConstants.signalEncoderModel

    # --------------------------- Model Parameters -------------------------- #

    # Initialize the model information classes.
    modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator)  # Initialize the model compiler.
    trainingProtocols = trainingProtocolHelpers(submodel=submodel, accelerator=accelerator)  # Initialize the training protocols.
    modelParameters = modelParameters(accelerator)  # Initialize the model parameters class.
    modelMigration = modelMigration(accelerator)  # Initialize the model migration class.

    # Specify training parameters
    trainingDate = modelCompiler.embedInformation(submodel, userInputParams, trainingDate)  # Embed training information into the name.
    numEpochs, numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo()  # The number of epochs to plot and save the model.
    datasetNames, metaDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    print("Arguments:", userInputParams)
    print(trainingDate, "\n")

    # Compile the final modules.
    allModels, allDataLoaders, allMetaModels, allMetadataLoaders, _ = modelCompiler.compileModelsFull(metaDatasetNames, submodel, testSplitRatio, datasetNames)
    allDataLoaders.append(allMetadataLoaders.pop(0))  # Do not metatrain with wesad data.
    datasetNames.append(metaDatasetNames.pop(0))  # Do not metatrain with wesad data.
    allModels.append(allMetaModels.pop(0))  # Do not metatrain with wesad data.

    # Do not train on the meta-datasets.
    if holdDatasetOut: allDataLoaders, datasetNames, allModels = [], [], []

    # Compile all the dataset names.
    allDatasetNames = metaDatasetNames + datasetNames

    # --------------------------- Figure Plotting -------------------------- #

    # Initialize plotting classes.
    saveFolder = os.path.dirname(os.path.abspath(__file__)) + "/finalFigures/"
    signalEncoderPlots = signalEncoderPlots(modelName, datasetNames, sharedModelWeights, savingBaseFolder=saveFolder, accelerator=accelerate.Accelerator(cpu=True))

