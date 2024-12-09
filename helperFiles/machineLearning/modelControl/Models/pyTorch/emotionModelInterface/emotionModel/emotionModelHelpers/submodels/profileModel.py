from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(profileModel, self).__init__()
        self.profileStatePath, self.profileOGStatePath, self.profileStateLosses = None, None, None
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments)
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments

        # Initialize the physiological profile.
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.physiologicalProfile = self.physiologicalInitialization(self.physiologicalProfile)
        self.profileStatePath, self.profileOGStatePath, self.profileStateLosses = [], [], []

    def getPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile.to(batchInds.device)[batchInds]
