# General
from natsort import natsorted
import torch
import os

# Import files.
from helperFiles.machineLearning.feedbackControl.heatAndMusicTherapy.helperMethods.dataInterface.dataInterface import dataInterface
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing.extractDataProtocols import extractData


class empatchProtocols(extractData):

    def __init__(self, predictionOrder, predictionBounds, modelParameterBounds, therapyExpMethod="HeatingPad"):
        super().__init__()
        # General parameters.
        self.modelParameterBounds = modelParameterBounds  # The bounds for the model parameters.
        self.predictionBounds = predictionBounds  # The bounds for the predictions.
        self.predictionOrder = predictionOrder  # The order of the predictions.
        self.therapyExpMethod = therapyExpMethod  # The therapy method to analyze.


        # Initialize helper classes.
        self.compileModelInfo = compileModelInfo()

        # Collected data parameters.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.getFeatureInformation()

        # Get the data folder.
        # self.trainingFolder = "/../../../../../../" + self.compileModelInfo.getTrainingDataFolder(useTherapyData=True)
        base_dir = os.path.dirname(__file__)
        relative_path = os.path.join(base_dir, "../../../../../../", self.compileModelInfo.getTrainingDataFolder(useTherapyData=True))
        self.trainingFolder = os.path.abspath(relative_path)

        # Assert the validity of the input parameters.
        baseTherapyMethod = self.therapyExpMethod.replace("OpenLoop", "").replace("ClosedLoop", "")
        self.compileModelInfo.assertValidTherapyMethod(therapyMethod=baseTherapyMethod)

    def getFeatureInformation(self):
        # Specify biomarker information.
        extractFeaturesFrom = self.compileModelInfo.streamingOrder  # A list with all the biomarkers from streamingOrder for feature extraction
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        return featureNames, biomarkerFeatureNames, biomarkerFeatureOrder

    def getTherapyData(self):
        # Initialize holders.
        stateHolder = []
        # For each file in the training folder.
        for excelFile in natsorted(os.listdir(self.trainingFolder)):
            # Remove any files that are not part of the therapy data.
            if not excelFile.endswith(".xlsx") or excelFile.startswith(("~", ".", "._")): continue  # Only analyze Excel files with the training signals.
            if self.therapyExpMethod not in excelFile: continue  # Only analyze the therapy data.
            print(f"Processing file: {excelFile}")

            # Get the full file information.
            savedFeaturesFile = self.trainingFolder + '/' + self.saveFeatureFolder + excelFile.split(".")[0] + self.saveFeatureFile_Appended
            # Extract the features from the Excel file.
            rawFeatureTimesHolder, rawFeatureHolder, _, experimentTimes, experimentNames, currentSurveyAnswerTimes, \
                currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                = self.getFeatures(self.biomarkerFeatureOrder, savedFeaturesFile, self.biomarkerFeatureNames, surveyQuestions=[], finalSubjectInformationQuestions=[])

            # currentSurveyAnswersList: The user answers to the survey questions during the experiment. Dimensions: numExperiments, numSurveyQuestions
            # surveyQuestions: The questions asked in the survey. Dimensions: numSurveyQuestions = numEmotionsRecorded
            # currentSurveyAnswerTimes: The times the survey questions were answered. Dimensions: numExperiments
            # Extract the mental health information.
            predictionOrder, finalLabels = self.compileModelInfo.extractFinalLabels(currentSurveyAnswersList, finalLabels=[])
            assert predictionOrder == self.predictionOrder, f"The given prediction order {predictionOrder} does not match the model's expected prediction order {self.predictionOrder}."
            # finalLabels: The final mental health labels for the experiment. Dimensions: numLabels, numExperiments

            finalLabelsTensor = torch.tensor(finalLabels)

            # Initialize a list to hold the normalized values
            normalizedValuesList = []
            # Loop through each column in finalLabelsTensor and normalize individually
            for i in range(finalLabelsTensor.shape[1]):
                currentParamValues = finalLabelsTensor[:, i]
                normalizedValues = dataInterface.normalizeParameters(
                    currentParamBounds=self.predictionBounds,
                    normalizedParamBounds=self.modelParameterBounds,
                    currentParamValues=currentParamValues
                )
                normalizedValuesList.append(normalizedValues)

            # Combine the normalized values back into a tensor and convert to list
            currentPredictionStates = torch.stack(normalizedValuesList, dim=1).tolist()

            def temperatureInterface():
                experimentTempHolder = []
                for experimentalInd in range(len(experimentTimes)):
                    experimentName = experimentNames[experimentalInd]
                    # Match experiments: "Heating-Baseline-temp" or "Baseline-HeatingPad-temp"
                    if "Heating" in experimentName or "HeatingPad" in experimentName:
                        experimentalTemp = int(experimentName.split("-")[-1])
                        experimentTempHolder.append(experimentalTemp)
                return experimentTempHolder

            def binauralBeatsInterface():
                experimentParamHolder = [[], []]
                for experimentalInd in range(len(experimentNames)):
                    experimentName = experimentNames[experimentalInd]
                    # match binaural beat experiment: "Binaural-Baseline-param1-param2"
                    if "Binaural" in experimentName:
                        parts = experimentName.split("-")
                        param1 = int(parts[-2])
                        param2 = int(parts[-1])
                        experimentParamHolder[0].append(param1)
                        experimentParamHolder[1].append(param2)
                return experimentParamHolder

            if self.therapyExpMethod == "HeatingPad":
                experimentTempHolder = temperatureInterface()
                heatingExpIndices = [i for i, name in enumerate(experimentNames) if "Heating" in name or "HeatingPad" in name]
                # For each experiment.
                for index, experimentalInd in enumerate(heatingExpIndices):
                    # Get the mental state values and normalize them.
                    emotion_states = [currentPredictionStates[0][experimentalInd], currentPredictionStates[1][experimentalInd], currentPredictionStates[2][experimentalInd]]
                    experimentalTemp = experimentTempHolder[index]
                    stateHolder.append([experimentalTemp] + emotion_states)
            elif self.therapyExpMethod == "BinauralBeats":
                experimentParamHolder = binauralBeatsInterface()
                correspondEmoIndex = [i for i, name in enumerate(experimentNames) if 'Binaural' in name]
                for index, experimentalInd in enumerate(correspondEmoIndex):
                    emotion_states = [currentPredictionStates[0][correspondEmoIndex[index]], currentPredictionStates[1][correspondEmoIndex[index]], currentPredictionStates[2][correspondEmoIndex[index]]]
                    experimentalParam_1 = experimentParamHolder[0][index]
                    experimentalParam_2 = experimentParamHolder[1][index]
                    stateHolder.append([experimentalParam_1, experimentalParam_2] + emotion_states)

        stateHolder = torch.as_tensor(stateHolder)

        if self.therapyExpMethod == "HeatingPad":
            parameter = stateHolder[:, 0].view(1, -1)
            pa = stateHolder[:, 1].view(1, -1)
            na = stateHolder[:, 2].view(1, -1)
            sa = stateHolder[:, 3].view(1, -1)
            return parameter, pa, na, sa
        elif self.therapyExpMethod == "BinauralBeats":
            parameter_1 = stateHolder[:, 0].view(1, -1)
            parameter_2 = stateHolder[:, 1].view(1, -1)
            totalParameter = [parameter_1, parameter_2]
            pa = stateHolder[:, 2].view(1, -1)
            na = stateHolder[:, 3].view(1, -1)
            sa = stateHolder[:, 4].view(1, -1)
            return totalParameter, pa, na, sa



