from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class inferenceModel(nn.Module):

    def __init__(self, encodedDimension):
        super(inferenceModel, self).__init__()
        self.encodedDimension = encodedDimension
        self.physiologicalProfile = None

    def resetInferenceModel(self, numExperiments, encodedDimension):
        self.physiologicalProfile = emotionModelWeights.getInitialPhysiologicalProfile(numExperiments)
        self.physiologicalProfile.requires_grad = True

    def getCurrentPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile[batchInds]
