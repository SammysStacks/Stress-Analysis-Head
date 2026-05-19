
import accelerate
import torch

import user_parameters
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers import modelConstants
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData


class adjustInputParameters:

    def __init__(self, deviceType, plotStreamedData=True, streamData=False, readDataFromExcel=False, trainModel=False, useModelPredictions=False, useTherapyData=False):
        # Set the parameters for the program.
        self.useModelPredictions = useModelPredictions or trainModel  # Use the Machine Learning Model for Predictions
        self.readDataFromExcel = readDataFromExcel  # Read Data from an Excel File
        self.compileModelInfo = compileModelInfo()  # Initialize the Model Information
        self.plotStreamedData = plotStreamedData  # Plot the Streamed Data in Real-Time
        self.useTherapyData = useTherapyData  # Use the Therapy Data for the Machine Learning Model
        self.streamData = streamData  # Stream data directly from a device.
        self.deviceType = deviceType  # The type of device being used.

    def getGeneralParameters(self):
        if self.deviceType == "empatica":
            # Specify biomarker information.
            streamingOrder = self.compileModelInfo.streamingOrder_e4  # A List Representing the Order of the Sensors being Streamed in: ["bvp", "acc", "eda", "temp"]
            extractFeaturesFrom = streamingOrder if self.useModelPredictions else []  # A list with all the biomarkers from streamingOrder for feature extraction
            allAverageIntervals = self.compileModelInfo.featureAverageWindows_e4  # acc: 30, bvp: 60, eda: 30, temp: 30
        else:
            # Specify biomarker information.
            streamingOrder = self.compileModelInfo.streamingOrder  # A List Representing the Order of the Sensors being Streamed in: ["eog", "eeg", "eda", "temp"]
            extractFeaturesFrom = streamingOrder  # if self.useModelPredictions else []  # A list with all the biomarkers from streamingOrder for feature extraction
            allAverageIntervals = self.compileModelInfo.featureAverageWindows  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60  Old: [120, 75, 90, 45]
        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        featureAverageWindows = []
        # Compile feature average windows.
        for biomarker in biomarkerFeatureOrder:
            featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])

        return streamingOrder, biomarkerFeatureOrder, featureAverageWindows, featureNames, biomarkerFeatureNames, extractFeaturesFrom

    def getSavingInformation(self, date, trialName, userName):
        # Specify the path to the collected data.
        collectedDataFolder = self.compileModelInfo.getTrainingDataFolder(self.useTherapyData)
        currentFilename = collectedDataFolder + f"{date} {trialName} Trial {userName}.xlsx"

        return collectedDataFolder, currentFilename

    def getStreamingParams(self, deviceAddress):
        # Arduino Streaming Parameters.
        voltageRange = (0, 3.3)
        adcResolution = 4096

        # Assert that you are using this protocol.
        if not self.streamData:
            return None, voltageRange, adcResolution, None, None

        # Streaming flags.
        recordQuestionnaire = not self.plotStreamedData  # Only use one GUI: questionnaire or streaming
        saveRawSignals = True  # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'

        return deviceAddress, voltageRange, adcResolution, saveRawSignals, recordQuestionnaire

    def getPlottingParams(self, analyzeBatches=False):
        # Analyze the data in batches.
        if self.deviceType == "serial":
            numPointsPerBatch = 4000  # The Number of Data Points to Display to the User at a Time.
            moveDataFinger = 400  # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;
        elif self.deviceType == "empatica":
            numPointsPerBatch = 50  # The Number of Data Points to Display to the User at a Time.
            moveDataFinger = 5  # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;
        else: raise ValueError(f"Unknown device type: {self.deviceType}")

        if not analyzeBatches:
            # If displaying all data, read in all the Excel data (max per sheet) at once
            numPointsPerBatch = 2048576
            moveDataFinger = 1048100

        return numPointsPerBatch, moveDataFinger

    def getExcelParams(self):
        # Assert that you are using this protocol.
        if not self.readDataFromExcel:
            return False, None

        # Specify the Excel Parameters.
        saveRawFeatures = False  # Save the Raw Features to an Excel File
        testSheetNum = 0  # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document

        return saveRawFeatures, testSheetNum

    def getMachineLearningParams(self):
        # Train or test the machine learning modules
        if not self.useModelPredictions:
            return None, [], None, None, None

        # Specify the Machine Learning Parameters
        plotTrainingData = False  # Plot all training information
        actionControl = None  # NOT IMPLEMENTED YET
        # If training, read the data as quickly as possible

        # Define the accelerator parameters.
        accelerator = accelerate.Accelerator(
            dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
            cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
            step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
            gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
            mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
        )

        # Load in the model.
        user_parameters.set_params()
        modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator, validationRun=True)  # Initialize the model compiler.
        modelClasses, dataLoaders, _, _, _ = modelCompiler.compileModelsFull(metaDatasetNames=[], submodel=modelConstants.modelConstants.emotionModel,
                                                                             testSplitRatio=0.2, datasetNames=[modelConstants.modelConstants.empatchDatasetName],
                                                                             loadSubmodelDate="2026-05-18")

        # Choose the Folder to Save ML Results
        saveModel = not self.useModelPredictions  # Save the Machine Learning Model for Later Use

        return modelClasses, actionControl, plotTrainingData, saveModel

    def getModelParameters(self):
        # Train or test the machine learning modules
        if not self.useModelPredictions:
            return None, None, None

        # Specify the MTG-Jamendo dataset path
        soundInfoFile = 'raw_30s_cleantags_50artists.tsv'
        dataFolder = './therapyHelperFiles/machineLearning/_Feedback Control/Music Therapy/Organized Sounds/MTG-Jamendo/'
        # Initialize the classes
        # soundManager = oldMusicTherapy.soundController(dataFolder, soundInfoFile) # Controls the music playing
        # soundManager.loadSound(soundManager.soundInfo[0][3])
        playGenres = [None, 'pop', 'jazz', 'heavymetal', 'classical', None]
        # playGenres = [None, 'hiphop', 'blues', 'disco', 'ethno', None]
        # playGenres = [None, 'funk', 'reggae', 'rap', 'classicrock', None]

        # playGenres = [None, 'hiphop', 'blues', 'hardrock', 'african', None]
        # soundManager.pickSoundFromGenres(playGenres)

        return soundInfoFile, dataFolder, playGenres
