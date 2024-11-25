from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation, self).__init__()
        self.physiologicalProfile = None

        # Initialize the blank signal profile.
        self.resetTrainingProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def resetTrainingProfile(self, numExperiments, encodedDimension):
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def getCurrentPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile[batchInds]
