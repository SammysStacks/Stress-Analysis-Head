import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation, self).__init__()
        self.encodedDimension = encodedDimension
        self.physiologicalProfileFFT = None

        # Initialize the blank signal profile.
        self.resetTrainingProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def resetTrainingProfile(self, numExperiments, encodedDimension):
        self.physiologicalProfileFFT = self.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def getCurrentPhysiologicalProfile(self, batchInds):
        physiologicalProfileFFT = self.physiologicalProfileFFT[batchInds]
        physiologicalProfile = torch.fft.irfft(physiologicalProfileFFT, n=self.encodedDimension, dim=-1, norm='ortho')

        return physiologicalProfile
