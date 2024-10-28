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

        self.timeEmoAnalysisWindow = modelConstants.timeWindows[-1]



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

    def compileAllFeatureWPadding(self, startTimePointer, timeEmoAnalysisWindow, featureTimes, features):
        # --------------- Get the max feature length for padding ----------------#
        # Find the maximum length for padding feature times
        maxSequenceLength = max(
            max(len(featureChannelTimes) for featureChannelTimes in biomarkerTimes)
            for biomarkerTimes in featureTimes
        )

        # Find the maximum length for padding features should be the same as maxSequenceLength
        maxFeatureLength = max(
            max(len(featureChannel) for featureChannel in biomarkerFeatures)
            for biomarkerFeatures in features
        )

        # Padding to have data structure: numNewPoints, numFeatures, maxSequenceLength, [time, compiled feature data]
        numNewPoints = startTimePointer - timeEmoAnalysisWindow

        # Calculate the total number of features (sum of all feature channels across biomarkers)
        numFeatures = sum(len(biomarkerFeatures) for biomarkerFeatures in features)

        # compiling features
        compiledAllFeatures = torch.zeros((numNewPoints, numFeatures, maxSequenceLength, len(modelConstants.signalChannelNames)), dtype=torch.float32)

        # --------------- Pad feature times and features ----------------#
        featureIdx = 0
        for biomarkerTimes, biomarkerFeatures in zip(featureTimes, features):
            for featureChannelTimes, featureChannel in zip(biomarkerTimes, biomarkerFeatures):
                # get the length index for filling in the values
                length = min(len(featureChannelTimes), len(featureChannel), maxSequenceLength)

                # Fill the time data
                compiledAllFeatures[:, featureIdx, :length, 0] = torch.tensor(featureChannelTimes[:length])

                # Fill the feature data
                compiledAllFeatures[:, featureIdx, :length, 1] = torch.tensor(featureChannel[:length])
                featureIdx += 1

        return compiledAllFeatures


    def predictLabels(self, featureTimes, features):

        # 1: Organize compile features like the raw features (average back)
        newCompiledFeatureTimes = 1  # similar to rawfeatures
        newCompiledFeatures = 1

        startTimePointer =

        # 2: Get a last unanalyzed points, and put them into a data structure: numPointUnAnalyzed, numFeatures, maxSequenceLength, [time, compiled feature data]
        # Times go from furtherest away -> 0 (the current time)

        # 3: train the inferewnce model
        _, _, _, _, _, emotionProfile = self.modelClasses[0].model.forward(allNewFeatures)
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



