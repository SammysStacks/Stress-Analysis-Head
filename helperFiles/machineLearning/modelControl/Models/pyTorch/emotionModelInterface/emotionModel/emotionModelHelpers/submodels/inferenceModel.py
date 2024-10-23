from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderComponents.emotionModelWeights import emotionModelWeights


class inferenceModel(nn.Module):

    def __init__(self, encodedDimension):
        super(inferenceModel, self).__init__()
        self.encodedDimension = encodedDimension
        self.physiologicalProfile = None

    def resetInferenceModel(self, numExperiments):
        self.physiologicalProfile = emotionModelWeights.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=self.encodedDimension)
        self.physiologicalProfile.requires_grad = True

    def getCurrentPhysiologicalProfile(self, batchInds):
        self.physiologicalProfile.requires_grad = True
        return self.physiologicalProfile[batchInds]
