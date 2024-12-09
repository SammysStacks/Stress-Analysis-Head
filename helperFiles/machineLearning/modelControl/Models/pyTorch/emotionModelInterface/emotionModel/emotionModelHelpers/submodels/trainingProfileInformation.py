from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation, self).__init__()
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments=numExperiments)
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.inferenceStatePath = None

        # Initialize the physiological profile.
        self.resetInferenceProfileWeights()

    def resetInferenceProfileWeights(self):
        self.physiologicalProfile = self.physiologicalInitialization(self.physiologicalProfile)
        self.inferenceStatePath = []

    def getInferencePhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile.to(batchInds.device)[batchInds]
