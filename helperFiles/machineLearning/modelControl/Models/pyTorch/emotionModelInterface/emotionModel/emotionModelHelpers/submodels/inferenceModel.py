from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class inferenceModel(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(inferenceModel, self).__init__()
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments=numExperiments)
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.resetInferenceProfile()

    def resetInferenceProfile(self):
        self.physiologicalProfile = self.physiologicalInitialization(self.physiologicalProfile)

    def getCurrentPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile.to(batchInds.device)[batchInds]
