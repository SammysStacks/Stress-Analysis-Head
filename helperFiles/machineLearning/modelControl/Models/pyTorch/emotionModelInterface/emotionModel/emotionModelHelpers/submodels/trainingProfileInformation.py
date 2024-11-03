from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation).__init__()
        # Initialize the blank signal profile.
        self.physiologicalProfileAnsatz = self.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def getCurrentPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfileAnsatz[batchInds]
