import os

# Import Files
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyMain import heatTherapyControl




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
        self.alignedFeatureLabels = None  # The FINAL predicted labels at the current timepoint.
        self.alignedFeatureTimes = None   # The interpolated timepoints of the ALIGNED feature.
        self.alignedFeatures = None       # Linearly interpolated features to align all at the same timepoint.
        self.alignedUserNames = []
        self.alignedItemNames = []
        self.userName = None
                
        # Initialize mutable variables.
        self.resetVariables_HMI()

    def therapyInitialization(self):
        assert self.actionControl in {"heat", "music", "chatGPT"}, f"Invalid actionControl: {self.actionControl}. Must be one of 'heat', 'music', or 'chatGPT'."
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
        # Aligned feature data structure
        self.alignedFeatureLabels = [[] for _ in range(len(self.modelClasses))]  # The FINAL predicted labels at the current timepoint.
        self.alignedFeatures = [[] for _ in range(len(self.featureNames))]       # Interpolated features to align all at the same timepoint. Dimensions: [numFeatures, numTimepoints]
        self.alignedFeatureTimes = []   # The interpolated timepoints of the ALIGNED feature. Dimensions: [numTimepoints]

        # Subject information
        self.alignedUserNames = []
        self.alignedItemNames = []
        self.userName = None
        
    def setUserName(self, filePath):
        # Get user information
        fileName = os.path.basename(filePath).split(".")[0]
        self.userName = fileName.split(" ")[-1].lower()

    def predictLabels(self): 
        # Find the new final features, where no label has been predicted yet
        allNewFeaturesTimes = self.alignedFeatureTimes[len(self.alignedFeatureLabels[0]):].copy()
        allNewFeatures = self.alignedFeatures[:, len(self.alignedFeatureLabels[0]):].copy()

        currentTimePoint = None
        currentParam = None
        currentPrediction = None
        self.therapyControl.therapyProtocol.updateEmotionPredState(self.userName, currentTimePoint, currentParam, currentPrediction)
        therapyState, allMaps = self.therapyControl.therapyProtocol.updateTherapyState()


        return therapyState



