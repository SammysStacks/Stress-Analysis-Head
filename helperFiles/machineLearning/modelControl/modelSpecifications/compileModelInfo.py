import json

import numpy as np
import torch
import math
import os

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters

class compileModelInfo:

    def __init__(self):
        # Store Extracted Features
        self.modelFolder = os.path.dirname(__file__) + "/_finalModels/"
        self.trainingDataFolder = self.getTrainingDataFolder(useTherapyData=False)

        # Specify the hardcoded strings.
        self.activityNames = np.asarray(["Baseline", "Music", "CPT", "Exercise", "VR", "Recovery"])
        self.therapyNames = np.asarray(["HeatingPad", "BinauralBeats", "Images", "Voice"])
        self.predictionOrder = ["Positive Affect", "Negative Affect", "State Anxiety"]
        self.streamingOrder_e4 = ["bvp", "acc", "eda", "temp"]
        self.streamingOrder = ["eog", "eeg", "eda", "temp"]

        # Hardcoded feature information.
        self.featureAverageWindows = [60, 30, 30, 30]  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60  Old: [120, 75, 90, 45]
        self.featureAverageWindows_e4 = [30, 60, 30, 30]  # Acc, Bvp, EDA, Temp
        self.newSamplingFreq = 1  # The new sampling frequency for the data.

        self.surveyInfoLocation = os.path.dirname(__file__) + "/../../../../helperFiles/surveyInformation/"
        # Specify the survey information.
        self.numQuestionOptions = [5] * 10  # PANAS Survey
        self.numQuestionOptions.extend([4] * 20)  # STAI Survey
        # Specify what each model is predicting
        self.predictionBounds = ((5, 25), (5, 25), (20, 80))
        self.predictionWeights = [0.1, 0.1, 0.8]
        self.optimalPredictions = (25, 5, 20)
        # Specify the order of the survey questions.
        self.posAffectInds = [2, 4, 6, 7, 8]
        self.negAffectInds = [0, 1, 3, 5, 9]
        self.staiInds_Pos = [10, 11, 14, 17, 19, 20, 24, 25, 28, 29]
        self.staiInds_Neg = [12, 13, 15, 16, 18, 21, 22, 23, 26, 27]
        # Specify the order of the survey questions.
        self.panasInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.staiInds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.allInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.numSurveyQuestions = len(self.posAffectInds) + len(self.negAffectInds) + len(self.staiInds_Pos) + len(self.staiInds_Neg)
        # Specify the survey question statistics.
        self.standardDeviationEmotions = torch.tensor([
            0.65865002, 0.62711051, 1.18120277, 0.4208254, 1.07318031,
            0.99532107, 1.02520504, 1.18671143, 1.12580164, 0.71014069,
            1.00852383, 0.91094593, 0.91744722, 0.86153846, 0.96780522,
            0.6831108, 0.9048921, 0.88772771, 0.50828766, 1.00172107,
            0.87294042, 0.77253989, 0.8787655, 0.6742371, 1.00745414,
            0.89999635, 0.75623205, 0.79835465, 0.96063205, 0.94781662
        ])
        self.standardDeviationEmotionsPA = self.standardDeviationEmotions[self.posAffectInds]
        self.standardDeviationEmotionsNA = self.standardDeviationEmotions[self.negAffectInds]
        self.standardDeviationEmotionsSTAI = self.standardDeviationEmotions[self.staiInds]

        # Specify the STAI statistics.
        self.internalConsistencyRangeSTAI = (0.86, 0.95)
        self.standardDeviationSTAI = 11.69218848
        self.internalConsistencySTAI = 0.9
        # Specify the PA statistics.
        self.internalConsistencyRangePA = (0.86, 0.9)
        self.standardDeviationPA = 4.1965704
        self.testRetestReliabilityPA = 0.79
        # Specify the PA statistics.
        self.internalConsistencyRangeNA = (0.84, 0.87)
        self.standardDeviationNA = 2.65962581
        self.testRetestReliabilityNA = 0.81

        # Calculate the standard error of measurement for the exams.
        self.standardErrorMeasurementSTAI = self.calculateStandardErrorMeasurement(self.standardDeviationSTAI, self.internalConsistencySTAI)
        self.standardErrorMeasurementPA = self.calculateStandardErrorMeasurement(self.standardDeviationPA, self.testRetestReliabilityPA)
        self.standardErrorMeasurementNA = self.calculateStandardErrorMeasurement(self.standardDeviationNA, self.testRetestReliabilityNA)
        # Compile the standard error of measurements in the correct order.
        self.standardErrorMeasurements = [self.standardErrorMeasurementPA, self.standardErrorMeasurementNA, self.standardErrorMeasurementSTAI]

        # Therapy Parameters
        # Heat Therapy
        self.userTherapyMethod = "aStarTherapyProtocol"
        self.therapyInitialPredictions = torch.full(size=(1, 3, 1, 1), fill_value=0.5)
        self.therapyStartTime = torch.tensor(0)
        self.parameterBounds = (30, 50)
        self.parameterBinWidth = 1.5

    def compileSurveyInformation(self):
        # Get the data from the json file
        with open(self.surveyInfoLocation + "PANAS Questions.json") as questionnaireFile:
            panasInfo = json.load(questionnaireFile)

        with open(self.surveyInfoLocation + "I-STAI-Y1 Questions.json") as questionnaireFile:
            staiInfo = json.load(questionnaireFile)

        surveyTitles = ["PANAS"]
        # Extract the questions, answerChoices, and surveyInstructions
        surveyQuestions = [panasInfo['questions']]
        surveyAnswerChoices = [panasInfo['answerChoices']]
        surveyInstructions = [panasInfo['surveyInstructions'][0]]

        surveyTitles.append("STAI")
        # Extract the questions, answerChoices, and surveyInstructions
        surveyQuestions.append(staiInfo['questions'])
        surveyAnswerChoices.append(staiInfo['answerChoices'])
        surveyInstructions.append(staiInfo['surveyInstructions'][0])

        return surveyTitles, surveyQuestions, surveyAnswerChoices, surveyInstructions

    def assertValidTherapyMethod(self, therapyMethod):
        assert therapyMethod in self.therapyNames, f"Invalid therapy method: {therapyMethod}"

    def getHeatingTherapyNames(self):
        return self.therapyNames[0]

    def getMusicTherapyNames(self):
        return self.therapyNames[1]

    def getImageTherapyNames(self):
        return self.therapyNames[2]

    def getVoiceTherapyNames(self):
        return self.therapyNames[3]

    def getUserInputParameters(self):

        userInputParams = {'submodel': 'signalEncoderModel', 'optimizerType': 'AdamW', 'reversibleLearningProtocol': 'rCNN', 'irreversibleLearningProtocol': 'FC', 'deviceListed': 'cpu', 'numSpecificEncoderLayers': 4, 'numSharedEncoderLayers': 8,
                    'encodedDimension': 256, 'operatorType': 'wavelet', 'waveletType': 'bior3.7', 'numBasicEmotions': 6, 'numActivityModelLayers': 8, 'numEmotionModelLayers': 8, 'numActivityChannels': 4}
        userInputParams = modelParameters.getNeuralParameters(userInputParams)
        return userInputParams

    @staticmethod
    def getTrainingDataFolder(useTherapyData):
        # Get the correct data folder name.
        dataFolderName = "_finalTherapyData" if useTherapyData else "_finalDataset"
        return f"./_experimentalData/_empatchDataset/{dataFolderName}/"

    @staticmethod
    def calculateStandardErrorMeasurement(standardDeviation, reliability):
        return standardDeviation * math.sqrt(1 - reliability)

    def compileModelPaths(self, modelFile, modelTypes):
        # Extract the model filename and extension.
        modelFilename, modelExtension = modelFile.split(".")

        modelPaths = []
        # For each prediction model
        for modelInd in range(len(self.predictionOrder)):
            predictionType = "_" + self.predictionOrder[modelInd].replace(" ", "")

            # Create a new path for each model
            finalModelPath = self.modelFolder + modelFilename + predictionType + "_" + modelTypes[modelInd] + "." + modelExtension
            modelPaths.append(finalModelPath)

        # Assert the integrity of the results.
        assert len(modelTypes) == len(modelPaths), f"Signed contract for {len(modelPaths)} models, but given {len(modelTypes)} models."
        return modelPaths

    # ---------------------------------------------------------------------- #
    # ----------------------- Label-Specific Methods ----------------------- #

    def extractFinalLabels(self, surveyAnswersList, finalLabels=()):
        assert len(surveyAnswersList[0]) == self.numSurveyQuestions
        # Configure the input variables to numpy.
        surveyAnswersList = np.asarray(surveyAnswersList)
        # Create holder for final labels
        if len(finalLabels) == 0:
            finalLabels = [[] for _ in range(len(self.predictionOrder))]

        # Score each medical survey.
        positiveAffectivity, negativeAffectivity = self.scorePANAS(surveyAnswersList)
        stateAnxiety = self.scoreSTAI(surveyAnswersList)
        # Store the survey results.
        finalLabels[0].extend(positiveAffectivity)
        finalLabels[1].extend(negativeAffectivity)
        finalLabels[2].extend(stateAnxiety)

        return self.predictionOrder, finalLabels

    def scorePANAS(self, surveyAnswersList):
        # Extract the psych scores from the survey answers.
        positiveAffectivity = surveyAnswersList[:, self.posAffectInds].sum(axis=1)
        negativeAffectivity = surveyAnswersList[:, self.negAffectInds].sum(axis=1)
        return positiveAffectivity, negativeAffectivity

    def scoreSTAI(self, surveyAnswersList):
        # Extract the psych score from the survey answers.
        negativeStateAnxiety = surveyAnswersList[:, self.staiInds_Neg].sum(axis=1)
        positiveStateAnxiety = (5 - surveyAnswersList[:, self.staiInds_Pos]).sum(axis=1)
        # positiveStateAnxiety = ( - surveyAnswersList[:, self.staiInds_Pos]).sum(axis=1)
        return positiveStateAnxiety + negativeStateAnxiety

    # ---------------------------------------------------------------------- #
    # --------------------- Experiment-Specific Methods -------------------- #

    def extractActivityInformation(self, experimentalOrder, distinguishBaselines=False):
        activityNames = self.getActivityNames(distinguishBaselines)

        activityLabels = []
        # For each experiment conducted.
        for experimentName in experimentalOrder:
            # Get the activity classification.
            activityName = self.getActivityName(experimentName, distinguishBaselines)
            activityClass = self.getActivityClass(activityName)
            # Store the final classification.
            activityLabels.append(activityClass)

        return activityNames, activityLabels

    def getActivityNames(self, distinguishBaselines=False):
        return self.activityNames[0:len(self.activityNames) - (not distinguishBaselines)]

    def labelExperimentalOrder(self, experimentalOrder, distinguishBaselines=False):

        labeledActivities = []
        # For each experiment conducted.
        for experimentInd in range(len(experimentalOrder)):
            experimentName = experimentalOrder[experimentInd]

            # Get the activity that was performed.
            activityName = self.getActivityName(experimentName, distinguishBaselines)

            # If the activity is Baseline.
            if activityName == "Baseline":
                activityGroup = self.getActivityName(experimentalOrder[experimentInd + 1], distinguishBaselines)
                activityName = activityGroup + " Baseline"
            elif activityName == "Recovery":
                activityGroup = self.getActivityName(experimentalOrder[experimentInd - 1], distinguishBaselines)
                activityName = activityGroup + " Recovery"
            else:
                activityName = activityName + " Activity"

            labeledActivities.append(activityName)

        return labeledActivities

    def getActivityClass(self, activityName):
        activityClass = int(np.where(self.activityNames == activityName)[0][0])
        return activityClass

    def getActivityName(self, experimentName, distinguishBaselines=False):
        # Music has the song hash.
        if experimentName.isdigit():
            activityName = "Music"

        # Recovery is just another baseline.
        elif "Recovery" == experimentName:
            if distinguishBaselines:
                activityName = experimentName
            else:
                activityName = "Baseline"

        # VR is labeled with what is shown.
        elif "VR" in experimentName.split(" - "):
            activityName = "VR"

        # At this point, call the activity what we labeled.
        elif experimentName not in self.activityNames:
            assert False, f"Unknown experiment name: {experimentName}"
        else:
            activityName = experimentName

        return activityName

# -------------------------------------------------------------------------- #
