import os
import collections
import torch
import numpy as np

# Import Files
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyMain import heatTherapyControl
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.dataInterface.compileModelDataHelpers import compileModelDataHelpers

class humanMachineInterface:
    
    def __init__(self, modelClasses, actionControl, extractFeaturesFrom):
        # Accelerator configuration steps
        submodel, userInputParams, accelerator = None, None, None
        self.compileModelHelpers = compileModelDataHelpers(submodel, userInputParams, accelerator)

        #TODO: not sure
        self.allSubjectInds = []

        # General parameters.
        self.actionControl = actionControl
        self.modelClasses = modelClasses        # A list of machine learning models.

        # Initialize helper classes.
        self.compileFeatureNames = compileFeatureNames()  # Initialize the Feature Information
        self.compileModelInfo = compileModelInfo()  # Initialize the Model Information

        # Therapy parameters
        self.timePointEvolution = []
        self.therapyControl = None
        self.therapyInitializedUser = False
        self.plottingTherapyIndicator = False
        self.therapyParam = 37  # TODO: this wont work with other therapies.
        self.therapyStartTime = torch.tensor(0)
        self.initialPredictions = torch.full(size=(1, 3, 1, 1), fill_value=0.5)
        self.therapyInitialization()

        # Compile the feature information.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(extractFeaturesFrom)

        # Holder parameters.
        self.therapyStates = None
        self.userName = None
                
        # Initialize mutable variables.
        self.resetVariables_HMI()

        # Holder parameters.
        self.rawFeatureTimesHolder = None  # A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
        self.rawFeaturePointers = None  # A list of pointers indicating the last seen raw feature index for each analysis and each channel.
        self.rawFeatureHolder = None  # A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures

        # predefined model time
        self.modelTimeWindow = modelConstants.timeWindows[-1]
        self.startModelTime = self.modelTimeWindow
        self.modelTimeBuffer = 1
        self.modelTimeGap = 20

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
        normalizedInputModelData = self.compileModelHelpers.normalizeSignals(inputModelData)
        allNumSignalPoints = torch.empty(size=(len(inputModelData[0]), len(self.featureNames)), dtype=torch.int)
        compiledNormalizedInputData = self.inputModelDataWithContextualInfo(normalizedInputModelData, allNumSignalPoints, dataInd=0)
        exit()
        _, _, _, _, _, _, emotionProfile = self.modelClasses[0].model.forward(compiledNormalizedInputData)
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

        currentTimePoint = modelTimes[-1] # make sure it is: timepoints: list of tensor: [tensor(0)]
        currentParam = therapyParam[-1] # make sure it is: list of tensor: [torch.Size([1, 1, 1, 1])
        currentPrediction = newMentalState[-1] # make sure it is: list of tensor: torch.Size([1, 3, 1, 1])
        self.therapyControl.therapyProtocol.updateEmotionPredState(self.userName, currentTimePoint, currentParam, currentPrediction)
        therapyState, allMaps = self.therapyControl.therapyProtocol.updateTherapyState()
        self.therapyStates.append(therapyState)
        
        # all the time points
        self.timePointEvolution = self.therapyControl.therapyProtocol.timepoints

        return therapyState, allMaps



