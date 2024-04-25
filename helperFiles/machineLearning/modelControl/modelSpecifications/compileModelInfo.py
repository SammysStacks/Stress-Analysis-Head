# Basic
import os
import numpy as np


class compileModelInfo:

    def __init__(self, modelFile="generalModel.pkl", modelTypes=[]):
        # Store Extracted Features
        self.modelFolder = os.path.dirname(__file__) + "/_finalModels/"
        # self.modelFolder = self.modelFolder
        self.modelFilename, self.modelExtension = modelFile.split(".")
        self.modelTypes = modelTypes

        # Specify what each model is predicting
        self.predictionOrder = ["Positive Affect", "Negative Affect", "State Anxiety"]
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

        # Specify the possible activities.
        self.activityNames = np.array(["Baseline", "Music", "CPT", "Exercise", "VR", "Recovery"])
        self.therapyNames = np.array(["Binaural", "Heating", "Images", "Voice"])

        # Assert data integrity across file logic.
        assert len(self.modelTypes) == len(self.predictionOrder), f"User trying to create {len(self.modelTypes)} models, but we expected the following models: {self.predictionOrder}"

    def compileModelPaths(self):
        modelPaths = []
        # For each prediction model
        for modelInd in range(len(self.predictionOrder)):
            predictionType = "_" + self.predictionOrder[modelInd].replace(" ", "")

            # Create a new path for each model
            finalModelPath = self.modelFolder + self.modelFilename + predictionType + "_" + self.modelTypes[modelInd] + "." + self.modelExtension
            modelPaths.append(finalModelPath)

        # Assert the integrity of the results.
        assert len(self.modelTypes) == len(modelPaths), f"Signed contract for {len(modelPaths)} models, but given {len(self.modelTypes)} models."
        return modelPaths

    # ---------------------------------------------------------------------- #
    # ----------------------- Label-Specific Methods ----------------------- #

    def extractFinalLabels(self, surveyAnswersList, finalLabels=[]):
        assert len(surveyAnswersList[0]) == self.numSurveyQuestions
        # Configure the inputs variables to numpy.
        surveyAnswersList = np.asarray(surveyAnswersList)
        # Create holder for final labels
        if len(finalLabels) == 0:
            finalLabels = [[] for _ in range(len(self.predictionOrder))]

        # Score each medical survey.
        positiveAffectiviy, negativeAffectiviy = self.scorePANAS(surveyAnswersList)
        stateAnxiety = self.scoreSTAI(surveyAnswersList)
        # Store the survey results.
        finalLabels[0].extend(positiveAffectiviy)
        finalLabels[1].extend(negativeAffectiviy)
        finalLabels[2].extend(stateAnxiety)

        return self.predictionOrder, finalLabels

    def scorePANAS(self, surveyAnswersList):
        # Extract the psych scores from the survey answers.
        positiveAffectiviy = surveyAnswersList[:, self.posAffectInds].sum(axis=1)
        negativeAffectiviy = surveyAnswersList[:, self.negAffectInds].sum(axis=1)
        return positiveAffectiviy, negativeAffectiviy

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

        # At this point, just call the activity what we labeled.
        elif experimentName not in self.activityNames:
            assert False, f"Unknown experiment name: {experimentName}"
        else:
            activityName = experimentName

        return activityName

# -------------------------------------------------------------------------- #
