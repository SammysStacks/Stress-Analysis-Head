import torch
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation, self).__init__()
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.physiologicalProfile = None
        self.downsizingRatio = 8

        # Initialize the blank signal profile.
        self.resetTrainingProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

    def resetTrainingProfile(self, numExperiments, encodedDimension):
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=encodedDimension, downsizingRatio=self.downsizingRatio)

    def getCurrentPhysiologicalProfile(self, batchInds):
        physiologicalProfile = self.physiologicalProfile[batchInds].unsqueeze(1)

        # Interpolate the physiological profile.
        physiologicalProfile = torch.nn.functional.interpolate(physiologicalProfile, size=None, scale_factor=self.downsizingRatio, mode='linear', align_corners=True, recompute_scale_factor=True, antialias=False)

        return physiologicalProfile.squeeze(1)


if __name__ == '__main__':
    # General parameters.
    _numExperiments, _encodedDimension = 10, 256

    # Initialize the training profile.
    trainingProfile = trainingProfileInformation(numExperiments=_numExperiments, encodedDimension=_encodedDimension)

    # Get the current physiological profile.
    _batchInds = torch.arange(start=0, end=_numExperiments, dtype=torch.long)
    _physiologicalProfile = trainingProfile.getCurrentPhysiologicalProfile(batchInds=_batchInds)

    # Plot the physiological profile.
    plt.plot(_physiologicalProfile[0].detach().cpu().numpy())
    plt.show()
