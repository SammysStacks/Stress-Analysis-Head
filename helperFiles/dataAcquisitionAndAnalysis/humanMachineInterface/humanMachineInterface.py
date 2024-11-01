import os
import accelerate
import torch

from helperFiles.machineLearning.dataInterface.compileModelDataHelpers import compileModelDataHelpers
# Import Files
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyMain import heatTherapyControl
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionPipeline import emotionPipeline
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo


class humanMachineInterface:

    def __init__(self, modelClasses, actionControl, extractFeaturesFrom):
        # General parameters.
        self.modelTimeWindow = modelConstants.timeWindows[-1]
        self.submodel = modelConstants.emotionModel
        self.startModelTime = self.modelTimeWindow
        self.actionControl = actionControl
        self.modelClasses = modelClasses  # A list of machine learning models.
        self.allSubjectInds = []

        # Therapy parameters
        self.timePointEvolution = []
        self.therapyControl = None
        self.therapyInitializedUser = False
        self.plottingTherapyIndicator = False
        self.therapyParam = 37  # TODO: this wont work with other therapies.
        self.therapyStartTime = torch.tensor(0)
        self.initialPredictions = torch.full(size=(1, 3, 1, 1), fill_value=0.5)
        self.therapyInitialization()

        # Hardcoded parameters.
        self.modelTimeBuffer = 1
        self.modelTimeGap = 20

        # Initialize helper classes.
        self.compileModelHelpers = compileModelDataHelpers(self.submodel, userInputParams={}, accelerator=None)
        self.compileFeatureNames = compileFeatureNames()  # Initialize the Feature Information
        self.compileModelInfo = compileModelInfo()  # Initialize the Model Information

        # Compile the feature information.
        _, self.surveyQuestions, _, _ = self.compileModelInfo.compileSurveyInformation()
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(extractFeaturesFrom)

        # ------------------------------ pipeline preparation for training ------------------------------
        accelerator = accelerate.Accelerator(
            dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
            cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
            step_scheduler_with_optimizer=True,  # Whether to wrap the optimizer in a scheduler.
            gradient_accumulation_steps=1,  # The number of gradient accumulation steps.
            mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
        )
        self.emoPipeline = emotionPipeline(accelerator, datasetName=None, allEmotionClasses=None, numSubjects=None, userInputParams={},
                                           emotionNames=self.surveyQuestions, activityNames=self.compileModelInfo.activityNames, featureNames=self.featureNames, submodel=self.submodel, numExperiments=6, reconstructionIndex=None)
        self.experimentalInds = torch.arange(0, 6, dtype=torch.int64)

        # Holder parameters.
        self.surveyAnswersList = None  # A list of lists of survey answers, where each element represents an answer to surveyQuestions.
        self.surveyAnswerTimes = None  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.experimentTimes = None  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = None  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).
        self.therapyStates = None
        self.userName = None

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
            initialTime = self.therapyStartTime
            initialParam = self.therapyControl.therapyProtocol.boundNewTemperature(self.therapyParam, bufferZone=0.01)
            initialPredictions = self.initialPredictions
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

    def inputModelDataWithContextualInfo(self, inputModelData, allNumSignalPoints, dataInd):
        numExperiments, numSignals, maxSequenceLength, numChannels = inputModelData.shape
        for experimentInd in range(numExperiments):
            self.allSubjectInds.append(0)

        numSignalIdentifiers = len(modelConstants.signalIdentifiers)
        numMetadata = len(modelConstants.metadata)

        # Create lists to store the new augmented data.
        compiledSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength + numSignalIdentifiers + numMetadata, numChannels))
        assert len(modelConstants.signalChannelNames) == numChannels

        # For each recorded experiment.
        for experimentInd in range(numExperiments):
            # Compile all the metadata information: dataset specific.
            subjectInds = torch.full(size=(numSignals, 1, numChannels), fill_value=self.allSubjectInds[experimentInd])
            datasetInds = torch.full(size=(numSignals, 1, numChannels), fill_value=dataInd)
            metadata = torch.hstack((datasetInds, subjectInds))
            # metadata dim: numSignals, numMetadata, numChannels

            # Compile all the signal information: signal specific.
            eachSignal_numPoints = allNumSignalPoints[experimentInd].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, numChannels)
            signalInds = torch.arange(numSignals).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, numChannels)
            batchInds = torch.full(size=(numSignals, 1, numChannels), fill_value=experimentInd)
            signalIdentifiers = torch.hstack((eachSignal_numPoints, signalInds, batchInds))
            # signalIdentifiers dim: numSignals, numSignalIdentifiers, numChannels

            # Assert the correct hardcoded dimensions.
            assert emotionDataInterface.getSignalIdentifierIndex(identifierName=modelConstants.numSignalPointsSI) == 0, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getSignalIdentifierIndex(identifierName=modelConstants.signalIndexSI) == 1, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getMetadataIndex(metadataName=modelConstants.datasetIndexMD) == 0, "Asserting I am self-consistent. Hardcoded assertion"
            assert emotionDataInterface.getMetadataIndex(metadataName=modelConstants.subjectIndexMD) == 1, "Asserting I am self-consistent. Hardcoded assertion"
            assert numSignalIdentifiers == 3, "Asserting I am self-consistent. Hardcoded assertion"
            assert numMetadata == 2, "Asserting I am self-consistent. Hardcoded assertion"

            # Add the demographic data to the feature array.
            compiledSignalData[experimentInd] = torch.hstack((inputModelData[experimentInd], signalIdentifiers, metadata))

        return compiledSignalData

    def predictLabels(self, modelTimes, inputModelData, therapyParam):
        # Add in contextual information to the data.
        allNumSignalPoints = torch.empty(size=(len(inputModelData[0]), len(self.featureNames)), dtype=torch.int)
        compiledInputData = self.inputModelDataWithContextualInfo(inputModelData, allNumSignalPoints, dataInd=0)
        dataLoader = zip(self.experimentalInds, compiledInputData)

        self.emoPipeline.trainModel(dataLoader, self.submodel, trainSharedLayers=False, inferenceTraining=True, profileTraining=False, numEpochs=10)
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