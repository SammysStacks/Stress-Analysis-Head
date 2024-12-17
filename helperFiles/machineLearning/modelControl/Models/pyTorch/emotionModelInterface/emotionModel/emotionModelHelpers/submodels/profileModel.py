from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class profileModel(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(profileModel, self).__init__()
        self.profileStatePath, self.profileOGStatePath, self.lastLayerStatePath = None, None, None
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments)
        self.profileStateLosses, self.compiledSignalEncoderLayerStatePath = None, None
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments

        # Initialize the physiological profile.
        self.resetProfileHolders()
        self.resetProfileWeights()

    def resetProfileWeights(self):
        self.physiologicalInitialization(self.physiologicalProfile)

    def resetProfileHolders(self):
        self.profileStatePath, self.profileOGStatePath, self.lastLayerStatePath = [], [], []
        self.profileStateLosses, self.compiledSignalEncoderLayerStatePath = [], []

    def getPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile.to(batchInds.device)[batchInds]
