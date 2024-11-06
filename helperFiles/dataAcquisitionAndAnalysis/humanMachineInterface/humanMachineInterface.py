import os
import accelerate
import torch

# Import Files
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionPipeline import emotionPipeline
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyMain import heatTherapyControl
from helperFiles.machineLearning.dataInterface.compileModelDataHelpers import compileModelDataHelpers


class humanMachineInterface:

    def __init__(self, modelClasses, actionControl, extractFeaturesFrom):
        # Initialize helper classes.
        accelerator = accelerate.Accelerator(
            dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
            cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
            step_scheduler_with_optimizer=True,  # Whether to wrap the optimizer in a scheduler.
            gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
            mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
        )

        # General parameters.
        self.modelTimeWindow = modelConstants.timeWindows[-1]
        self.submodel = modelConstants.emotionModel
        self.startModelTime = self.modelTimeWindow
        self.actionControl = actionControl
        self.modelClasses = modelClasses  # A list of machine learning models.
        self.allSubjectInds = []

        # Holder parameters.
        self.surveyAnswersList = None  # A list of lists of survey answers, where each element represents an answer to surveyQuestions.
        self.surveyAnswerTimes = None  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.experimentTimes = None  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = None  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).
        self.therapyStates = None
        self.userName = None

        # Hardcoded parameters.
        self.modelTimeBuffer = 1
        self.modelTimeGap = 20

        # model helpers intialization
        self.compileFeatureNames = compileFeatureNames()  # Initialize the Feature Information
        self.compileModelInfo = compileModelInfo()  # Initialize the Model Information
        self.compileModelHelpers = compileModelDataHelpers(self.submodel, userInputParams=self.compileModelInfo.getUserInputParameters(), accelerator=accelerator)

        # Compile the feature information.
        _, self.surveyQuestions, _, _ = self.compileModelInfo.compileSurveyInformation()
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(extractFeaturesFrom)

        # ------------------------------ pipeline preparation for training ------------------------------
        self.emoPipeline = emotionPipeline(accelerator, datasetName=None, allEmotionClasses=self.compileModelInfo.numQuestionOptions, numSubjects=1, userInputParams=self.compileModelInfo.getUserInputParameters(),
                                           emotionNames=self.compileModelInfo.numQuestionOptions, activityNames=self.compileModelInfo.activityNames, featureNames=self.featureNames, submodel=self.submodel, numExperiments=6,
                                           reconstructionIndex=None)
        self.trainingProtocols = trainingProtocolHelpers(submodel=self.submodel, accelerator=accelerator)  # Initialize the training protocols.
        self.experimentalInds = torch.arange(0, 6, dtype=torch.int64)

        # Therapy parameters
        self.therapySwitcher = 'heat'  # Example: 'music' or 'Vr'
        self.therapyParam = modelConstants.therapyParams.get(self.therapySwitcher)
        self.initialPredictions = self.compileModelInfo.therapyInitialPredictions
        self.therapyStartTime = self.compileModelInfo.therapyStartTime
        self.plottingTherapyIndicator = False
        self.therapyInitializedUser = False
        self.timePointEvolution = []
        self.therapyControl = None
        self.therapyInitialization()

        # Initialize mutable variables.
        self.resetVariables_HMI()

    def therapyInitialization(self):
        assert self.actionControl in {"heat", "music", "chatGPT", None}, f"Invalid actionControl: {self.actionControl}. Must be one of 'heat', 'music', or 'chatGPT'."
        if self.actionControl == 'heat' and self.userName != self.therapyInitializedUser:
            protocolParameters = {
                'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
                'numSimulationHeuristicSamples': 50,  # The number of simulation samples to generate.
                'numSimulationTrueSamples': 30,  # The number of simulation samples to generate.
                'simulateTherapy': False,  # Whether to simulate the therapy.
            }

            parameterBounds = self.compileModelInfo.parameterBounds
            parameterBinWdith = self.compileModelInfo.parameterBinWidth
            self.therapyControl = heatTherapyControl(self.userName, parameterBounds, parameterBinWdith, protocolParameters, therapyMethod=self.compileModelInfo.userTherapyMethod, plotResults=self.plottingTherapyIndicator)
            initialParam = self.therapyControl.therapyProtocol.boundNewTemperature(self.therapyParam, bufferZone=0.01)
            initialPredictions = self.initialPredictions
            initialTime = self.therapyStartTime
            self.therapyControl.therapyProtocol.initializeUserState(self.userName, initialTime, initialParam, initialPredictions)
            self.therapyInitializedUser = self.userName
        elif self.actionControl == 'music' and self.userName != self.therapyInitializedUser:
            pass
        else:
            pass

    def resetVariables_HMI(self):
        # Subject information
        self.therapyStates = []
        self.userName = None

        # Survey Information
        self.surveyAnswersList = []  # A list of lists of survey answers, where each element represents a list of answers to surveyQuestions.
        self.surveyAnswerTimes = []  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = []  # A list of survey questions, where each element in surveyAnswersList corresponds to this question order.

        # Experimental information
        self.experimentTimes = []  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = []  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).


    def setUserName(self, filePath):
        # Get user information
        fileName = os.path.basename(filePath).split(".")[0]
        self.userName = fileName.split(" ")[-1].lower()

    def emotionConversion(self, emotionScores):
        emotionScores[:10] = 0.5 + emotionScores[:10] / (5.5 - 0.5)
        emotionScores[10:] = 0.5 + emotionScores[10:] / (4.5 - 0.5)
        return emotionScores

    def predictLabels(self, modelTimes, inputModelData, allNumSignalPoints, therapyParam):
        # Add in contextual information to the data.
        allSubjectInds = torch.zeros(inputModelData.shape[0])  # torch.Size([75, 81, 72, 2]) batchSize, numSignals, maxLength, [time, features]
        datasetInd = 1
        compiledInputData = self.compileModelHelpers.addContextualInfo(inputModelData, allNumSignalPoints, allSubjectInds, datasetInd)
        self.trainingProtocols.inferenceTraining([compiledInputData], self.emoPipeline, self.submodel, compiledInputData, 256, numEpochs=20)
        exit()
        _, _, _, _, _, _, emotionProfile = self.modelClasses[0].model.forward(compiledInputData)
        # emotionProfile dim: numNewPoints, numEmotions=30, encodedDimension=256
        emotionProfile = emotionProfile.detach().cpu().numpy()

        # Get the emotion scores.
        emotionScores = emotionProfile.argmax(axis=-1)  # dim: numNewPoints, numEmotions
        newMentalState = []
        therapyState = []
        for newPoints in range(len(emotionScores)):
            # Convert the index to a score. First ten are 0.5-5.5 and next 20 are 0.5-4.5
            emotionScores[newPoints] = self.emotionConversion(emotionScores[newPoints])

            # Handle PANAS and STAI EMOTIONS score conversion seperately
            positiveAffectivity, negativeAffectivity = self.compileModelInfo.scorePANAS(emotionScores)
            stateAnxiety = self.compileModelInfo.scoreSTAI(emotionScores)

            mental_state_tensor = torch.tensor([positiveAffectivity, negativeAffectivity, stateAnxiety]).view(1, 3, 1, 1)
            newMentalState.append(mental_state_tensor)
            therapyState.append(therapyParam)
            # newMentalStates: numNewPoints, numMentalStates=3
            # therapyStates: numNewPoints, numTherapyInfo

        # Pass the output to the therapy model
        currentTimePoint = modelTimes[-1]  # make sure it is: timepoints: list of tensor: [tensor(0)]
        currentParam = therapyParam[-1]  # make sure it is: list of tensor: [torch.Size([1, 1, 1, 1])
        currentPrediction = newMentalState[-1]  # make sure it is: list of tensor: torch.Size([1, 3, 1, 1])
        self.therapyControl.therapyProtocol.updateEmotionPredState(self.userName, currentTimePoint, currentParam, currentPrediction)
        therapyState, allMaps = self.therapyControl.therapyProtocol.updateTherapyState()
        self.therapyStates.append(therapyState)

        # all the time points
        self.timePointEvolution = self.therapyControl.therapyProtocol.timepoints
        return therapyState, allMaps
