""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import os
import sys
import threading

# General
import numpy as np

from helperFiles.machineLearning.featureAnalysis.featurePlotting import featurePlotting

# Compiler flags.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import helper files.
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing import extractDataProtocols, saveDataProtocols  # Import interfaces for reading/writing data
from helperFiles.surveyInformation.questionaireGUI import stressQuestionnaireGUI  # Import file for GUI control
from helperFiles.dataAcquisitionAndAnalysis import streamingProtocols  # Import interfaces for reading/writing data
from helperFiles.machineLearning import trainingProtocols  # Import interfaces for reading/writing data
from adjustInputParameters import adjustInputParameters  # Import the class to adjust the input parameters

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Protocol switches: only the first true variably executes.
    readDataFromExcel = False  # For SINGLE FILE analysis. Analyze Data from Excel File called 'currentFilename' on Sheet Number 'testSheetNum'
    trainModel = False  # Train Model with ALL Data in 'collectedDataFolder'.
    streamData = True  # Stream in Data from the Board and Analyze.

    # User options during the run: any number can be true.
    useModelPredictions = False or trainModel  # Apply the learning algorithm to decode the signals.
    plotStreamedData = True  # Graph the data to show incoming signals.
    useTherapyData = True  # Use the Therapy Data folder for any files.

    # General program flags.
    reanalyzeData = False  # Reanalyze training files: don't use saved features
    reverseOrder = False  # Reverse the order of the data for training.

    # Specify the user parameters.
    userName = "Subject XYZ".replace(" ", "")
    trialName = "Trial Type"  # Experiment Type: Music ....
    date = "yyyy-mm-dd"

    # Specify experimental parameters.
    deviceAddress = '12ba4cb61c85ec11bc01fc2b19c2d21c'  # Board's Serial Number (port.serial_number). Only used if streaming data, else it gets reset to None.
    stopTimeStreaming = 60*100  # If Float/Int: The Number of Seconds to Stream Data; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
    deviceType = 'serial'  # The type of device being used for streaming.

    # ---------------------------------------------------------------------- #

    # Assert the proper use of the program
    assert sum((readDataFromExcel, streamData, trainModel)) == 1, "Only one protocol can be be executed."
    assert deviceType in ['empatica', 'serial'], "The device type must be either 'empatica' or 'serial'."

    # Define helper classes.
    saveInputs = saveDataProtocols.saveExcelData()
    inputParameterClass = adjustInputParameters(deviceType=deviceType, plotStreamedData=plotStreamedData, streamData=streamData, readDataFromExcel=readDataFromExcel,
                                                trainModel=trainModel, useModelPredictions=useModelPredictions, useTherapyData=useTherapyData)

    # Get the reading/saving information.
    numPointsPerBatch, moveDataFinger = inputParameterClass.getPlottingParams(analyzeBatches=not readDataFromExcel)
    collectedDataFolder, currentFilename = inputParameterClass.getSavingInformation(date, trialName, userName)

    # Compile all the protocol information.
    streamingOrder, biomarkerFeatureOrder, featureAverageWindows, featureNames, biomarkerFeatureNames, extractFeaturesFrom = inputParameterClass.getGeneralParameters()
    performMachineLearning, modelClasses, actionControl, plotTrainingData, saveModel = inputParameterClass.getMachineLearningParams(featureNames, collectedDataFolder)
    deviceAddress, voltageRange, adcResolution, saveRawSignals, recordQuestionnaire = inputParameterClass.getStreamingParams(deviceAddress)
    soundInfoFile, dataFolder, playGenres = inputParameterClass.getModelParameters()
    saveRawFeatures, testSheetNum = inputParameterClass.getExcelParams()

    # Initialize instance to analyze the data
    readData = streamingProtocols.streamingProtocols(deviceType, deviceAddress, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, extractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData)

    # ----------------------------- Stream the Data from circuit board ----------------------------- #

    if streamData:
        if not recordQuestionnaire:
            # Stream in the data from the circuit board
            readData.streamWearableData(adcResolution, stopTimeStreaming, currentFilename)
        else:
            # Stream in the data from the circuit board
            streamingThread = threading.Thread(target=readData.streamWearableData, args=(adcResolution, stopTimeStreaming, currentFilename), daemon=True)
            streamingThread.start()

            # Open the questionnaire GUI.
            folderPath = "./helperFiles/surveyInformation/"
            stressQuestionnaire = stressQuestionnaireGUI(readData, folderPath)

            # When the streaming stops, close the GUI/Thread.
            stressQuestionnaire.finishedRun()
            streamingThread.join()

    # ------------------------ ReStream a Single Excel File ------------------------ #

    elif readDataFromExcel:
        # Collect the Data from Excel
        compiledRawData_eachFreq, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions = \
            extractDataProtocols.extractData().getData(currentFilename, deviceType, streamingOrder=streamingOrder, testSheetNum=testSheetNum)
        # Empatica: compiledRawData = [[[T1], ... [Tn]], [[biomarker1], [biomarkerN]]]
        # Serial: compiledRawData = [[T1], [[biomarker1], [biomarkerN]]]

        streamingIndex = 0
        # Analyze the Data using the Correct Protocol
        for compiledRawDataInd in range(len(compiledRawData_eachFreq)):
            numStreamingSignals = len(compiledRawData_eachFreq[compiledRawDataInd][1])
            startStreamInd, endStreamInd = streamingIndex, streamingIndex + numStreamingSignals
            streamingIndex = endStreamInd

            newStreamingOrder = streamingOrder[startStreamInd:endStreamInd]
            newExtractFeaturesFrom = [item for item in extractFeaturesFrom if item in newStreamingOrder]

            readData = streamingProtocols.streamingProtocols(deviceType, deviceAddress, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, newStreamingOrder, newExtractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData)
            readData.streamExcelData(compiledRawData_eachFreq[compiledRawDataInd], experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, currentFilename)

    # ----------------------------- Extract Feature Data ----------------------------- #

    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Initializing the training class.
        trainingInterface = trainingProtocols.trainingProtocols(deviceType, biomarkerFeatureNames, streamingOrder, biomarkerFeatureOrder, collectedDataFolder, readData)

        checkFeatureWindow_EEG = False
        if checkFeatureWindow_EEG:
            featureTimeWindows = np.arange(5, 25, 5)
            # # featureTimeWindows = [5, 30, 60, 90, 120, 150, 180]
            excelFile = collectedDataFolder + '2022-12-16 Full Dataset TV.xlsx'
            trainingInterface.varyAnalysisParam(excelFile, featureAverageWindows, featureTimeWindows)

        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = trainingInterface.streamTrainingData(featureAverageWindows, plotTrainingData=plotTrainingData, reanalyzeData=reanalyzeData, metaTraining=False, reverseOrder=reverseOrder)

        # Assert the validity of the feature extraction
        for analysisInd in range(len(allRawFeatureHolders[0])):
            assert len(allRawFeatureHolders[0][analysisInd][0]) == len(biomarkerFeatureNames[analysisInd]), "Incorrect number of raw features extracted"
        print("\nFinished Feature Extraction")

        # plotting the stress score labels for different experiments
        featurePlotting.stressLabelPlotting(experimentalOrder, subjectOrder, featureLabelTypes, allFinalLabels)

    # ------------------ Extract Data into this Namespace ------------------ #

    if streamData or readDataFromExcel:
        # Extract the data
        tempReadings = np.asarray(readData.analysisProtocols['temp'].channelData if readData.analysisProtocols['temp'] is not None else [])
        eogReadings = np.asarray(readData.analysisProtocols['eog'].channelData if readData.analysisProtocols['eog'] is not None else [])
        eegReadings = np.asarray(readData.analysisProtocols['eeg'].channelData if readData.analysisProtocols['eeg'] is not None else [])
        edaReadings = np.asarray(readData.analysisProtocols['eda'].channelData if readData.analysisProtocols['eda'] is not None else [])
        timepoints = np.array(readData.analysisList[0].timepoints)  # Assuming each analysis has the same time points.

        # Extract the feature labels.
        surveyAnswersList = np.asarray(readData.surveyAnswersList)  # A list of feature labels at each instance.
        surveyAnswerTimes = np.asarray(readData.surveyAnswerTimes)  # A list of times associated with each feature label.
        surveyQuestions = np.asarray(readData.surveyQuestions)  # A list of the survey questions asked the user.

        # Extract the experiment information
        experimentTimes = np.asarray(readData.experimentTimes)
        experimentNames = np.asarray(readData.experimentNames)

        # Extract subject information
        subjectInformationQuestions = np.asarray(readData.subjectInformationQuestions)
        subjectInformationAnswers = np.asarray(readData.subjectInformationAnswers)

        # -------------------------- Save Input data --------------------------- #

        # Save the Data in Excel
        if saveRawSignals:

            # Double Check to See if a User Wants to Save the Data
            verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
            if verifiedSave.upper() != "Y":
                verifiedSave = input(f"\tI recorded your answer as {verifiedSave}. You really do not want to save ... One more time (Y/N): ")

            if verifiedSave.upper().replace(" ", "") == "Y":
                # Get the streaming data
                streamingTimes, streamingData = [], []
                for analysis in readData.analysisList:
                    streamingTimes.append(np.asarray(analysis.timepoints))
                    for analysisChannelInd in range(len(analysis.channelData)):
                        streamingData.append(np.asarray(analysis.channelData[analysisChannelInd]))

                # Initialize Class to Save the Data and Save
                saveInputs.saveData(deviceType, streamingTimes, streamingData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions,
                                    subjectInformationAnswers, subjectInformationQuestions, streamingOrder, currentFilename)
            else:
                print("User Chose Not to Save the Data")
        elif saveRawFeatures:
            # Initialize Class to Save the Data and Save
            saveInputs.saveRawFeatures(readData.rawFeatureTimesHolder, readData.rawFeatureHolder, biomarkerFeatureNames, biomarkerFeatureOrder, experimentTimes,
                                       experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, currentFilename)

    # ----------------------------- End of Program ----------------------------- #
