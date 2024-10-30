import os
import collections
import torch
import numpy as np

# Import Files
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyMain import heatTherapyControl
from helperFiles.dataAcquisitionAndAnalysis.humanMachineInterface import featureOrganization


class humanMachineInterface:
    
    def __init__(self, modelClasses, actionControl, extractFeaturesFrom):
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

    def minMaxScale(self, compiledAllFeatures):
        # inputModelData = torch.zeros((numNewPoints, len(self.featureNames), maxSequenceLength, len(modelConstants.signalChannelNames)), dtype=torch.float64)
        # inputModelData dim: numNewTimePoints, numBiomarkerFeatures, maxSequenceLength, numChannels
        featureMin = compiledAllFeatures.min(dim=(0, 2, 3), keepdim=True) # get the feature min except at the feature dimension ?
        featureMax = compiledAllFeatures.max(dim=(0, 2, 3), keepdim=True)

        pass

    def emotionConversion(self, emotionScores):

        emotionScores[:10] = 0.5 + emotionScores[:10] / (5.5 - 0.5)
        emotionScores[10:] = 0.5 + emotionScores[10:] / (4.5 - 0.5)
        return emotionScores

    def predictLabels(self, modelTimes, inputModelData, therapyParam):
        # Add in contextual information to the data.

        _, _, _, _, _, _, emotionProfile = self.modelClasses[0].model.forward(inputModelData)
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



