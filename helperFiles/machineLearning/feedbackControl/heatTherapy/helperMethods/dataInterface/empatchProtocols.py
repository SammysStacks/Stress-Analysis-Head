# General
import os

import numpy as np
from natsort import natsorted

from helperFiles.dataAcquisitionAndAnalysis.excelProcessing.extractDataProtocols import extractData
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo


class empatchProtocols(extractData):

    def __init__(self, predictionOrder):
        super().__init__()
        # General parameters.
        self.trainingFolder = os.path.dirname(__file__) + "/../../../../../../_experimentalData/allSensors/_finalTherapyData/"
        self.predictionOrder = predictionOrder

        # Collected data parameters.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerOrder = self.getFeatureInformation()

        # Initialize helper classes.
        self.modelInfoClass = compileModelInfo("_.pkl", [0, 1, 2])

    @staticmethod
    def getFeatureInformation():
        # Specify biomarker information.
        extractFeaturesFrom = ["eog", "eeg", "eda", "temp"]  # A list with all the biomarkers from streamingOrder for feature extraction
        featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        return featureNames, biomarkerFeatureNames, biomarkerOrder

    # ------------------------ Data Collection Methods ------------------------ #
    @staticmethod
    def stateNormalization(PA, NA, SA):
        # Normalize the state values.
        PA_norm = (PA - 5) / 20
        NA_norm = (NA - 5) / 20
        SA_norm = (SA - 20) / 60

        return [PA_norm, NA_norm, SA_norm]

    def getTherapyData(self):
        # Initialize holders.
        stateHolder = []  # The state values for each experiment. Dimensions: numExperiments, (T, PA, NA, SA); 2D array

        # For each file in the training folder.
        for excelFile in natsorted(os.listdir(self.trainingFolder)):
            # Only analyze Excel files with the training signals.
            if not excelFile.endswith(".xlsx") or excelFile.startswith(("~", ".")):
                continue
            # Only analyze the heating therapy data.
            if "HeatingPad" not in excelFile:
                continue
            # Get the full file information.
            savedFeaturesFile = self.trainingFolder + self.saveFeatureFolder + excelFile.split(".")[0] + self.saveFeatureFile_Appended
            print(savedFeaturesFile)

            # Extract the features from the Excel file.
            rawFeatureTimesHolder, rawFeatureHolder, _, experimentTimes, experimentNames, currentSurveyAnswerTimes, \
                currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                = self.getFeatures(self.biomarkerOrder, savedFeaturesFile, self.biomarkerFeatureNames, surveyQuestions=[], finalSubjectInformationQuestions=[])
            # currentSurveyAnswersList: The user answers to the survey questions during the experiment. Dimensions: numExperiments, numSurveyQuestions
            # surveyQuestions: The questions asked in the survey. Dimensions: numSurveyQuestions = numEmotionsRecorded
            # currentSurveyAnswerTimes: The times the survey questions were answered. Dimensions: numExperiments

            # Extract the mental health information.
            predictionOrder, finalLabels = self.modelInfoClass.extractFinalLabels(currentSurveyAnswersList, finalLabels=[])
            assert predictionOrder == self.predictionOrder, f"Expected prediction order: {self.predictionOrder}, but got {predictionOrder}"
            # print('rawfeatureTimeholder', rawFeatureTimesHolder[3])
            mean_temperature_list = np.asarray(rawFeatureHolder[3])[:, 0]
            # print('rawfeatureholder', mean_temperature_list)

            # Get the temperatures in the time window.
            allTemperatureTimes = np.asarray(rawFeatureTimesHolder[3])
            allTemperatures = np.asarray(mean_temperature_list)

            # For each experiment.
            for experimentalInd in range(len(experimentNames)):
                # Get the start and end times for the experiment.
                surveyAnswerTime = currentSurveyAnswerTimes[experimentalInd]
                startExperimentTime = experimentTimes[experimentalInd][0]
                experimentName = experimentNames[experimentalInd]

                # Extract the temperature used in the experiment.
                if "Heating" in experimentName:
                    experimentalTemp = int(experimentName.split("-")[-1])
                else:
                    # Get the temperature at the start and end of the experiment.
                    startTemperatureInd = np.argmin(np.abs(allTemperatureTimes - startExperimentTime))
                    surveyAnswerTimeInd = np.argmin(np.abs(allTemperatureTimes - surveyAnswerTime))

                    # Find the average temperature between the start and end times.
                    experimentalTemp = np.mean(allTemperatures[startTemperatureInd:surveyAnswerTimeInd])

                # Store the state values.
                emotion_states = self.stateNormalization(finalLabels[0][experimentalInd], finalLabels[1][experimentalInd], finalLabels[2][experimentalInd])
                stateHolder.append([experimentalTemp] + emotion_states)

        return np.asarray(stateHolder)













