import torch
from matplotlib import pyplot as plt

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.emotionModelWeights import emotionModelWeights


class trainingProfileInformation(emotionModelWeights):

    def __init__(self, numExperiments, encodedDimension):
        super(trainingProfileInformation, self).__init__()
        self.physiologicalProfile = self.getInitialPhysiologicalProfile(numExperiments=numExperiments)
        self.encodedDimension = encodedDimension
        self.numExperiments = numExperiments
        self.resetProfile()

    def resetProfile(self):
        self.physiologicalProfile = self.physiologicalInitialization(self.physiologicalProfile)

    def getCurrentPhysiologicalProfile(self, batchInds):
        return self.physiologicalProfile[batchInds]


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
