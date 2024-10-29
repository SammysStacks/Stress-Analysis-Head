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
        self.therapyControl = None
        self.therapyInitializedUser = False
        self.plottingTherapyIndicator = False
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
                'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
                'numSimulationHeuristicSamples': 50,  # The number of simulation samples to generate.
                'numSimulationTrueSamples': 30,  # The number of simulation samples to generate.
                'simulateTherapy': False,  # Whether to simulate the therapy.
            }

            parameterBounds = self.compileModelInfo.parameterBounds
            parameterBinWdith = self.compileModelInfo.parameterBinWidth
            self.therapyControl = heatTherapyControl(self.userName, parameterBounds, parameterBinWdith, protocolParameters, therapyMethod=self.compileModelInfo.userTherapyMethod, plotResults=self.plottingTherapyIndicator)
            self.therapyControl.therapyProtocol.initializeUserState(self.userName)
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


    def predictLabels(self, compiledAllFeatures):

        _, _, _, _, _, emotionProfile = self.modelClasses[0].model.forward(compiledAllFeatures)
        self.startModelTime += self.moveDataFinger
        # emotionProfile dim: numNewPoints, numEmotions=30, encodedDimension=256

        # Get the emotion scores.
        emotionScores = emotionProfile.argmax(dim=-1)
        # Convert the index to a score. First ten are 0.5-5.5 and next 20 are 0.5-4.5
        # Handle PANAS and STAI EMOTIONS score conversion seperately

        # Now convert to PANAS and STAI
        newMentalStates = 1
        # newMentalStates: numNewPoints, numMentalStates=3
        # therapyStates: numNewPoints, numTherapyInfo

        # 4: put the output into the therapy

        currentTimePoint = None
        currentParam = None
        currentPrediction = None
        self.therapyControl.therapyProtocol.updateEmotionPredState(self.userName, currentTimePoint, currentParam, currentPrediction)
        therapyState, allMaps = self.therapyControl.therapyProtocol.updateTherapyState()


        return therapyState



