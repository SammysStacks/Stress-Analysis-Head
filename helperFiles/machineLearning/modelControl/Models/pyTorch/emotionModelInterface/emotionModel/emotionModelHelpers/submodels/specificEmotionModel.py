# PyTorch
import torch
from torch import nn


class specificEmotionModel(nn.Module):

    def __init__(self, numCommonSignals, numActivityFeatures, activityNames, numBasicEmotions, numInterpreterHeads, numSubjects, emotionNames, featureNames):
        super(specificEmotionModel, self).__init__()
        # General model parameters.
        self.numActivityFeatures = numActivityFeatures  # The number of common activity features to extract.
        self.numInterpreterHeads = numInterpreterHeads  # The number of ways to interpret a set of physiological signals.
        self.numCommonSignals = numCommonSignals  # The number of features from considering all the signals.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions (basis states of emotions).
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.

        # ------------------------------------------------------------------ # 

        # Reset the model
        self.resetModel()

    def forward(self):
        return None

        # ------------------------------------------------------------------ #  

    # DEPRECATED
    def shapInterface(self, reshapedSignalFeatures):
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numFeatures = reshapedSignalFeatures.shape
        reshapedSignalFeatures = torch.tensor(reshapedSignalFeatures.tolist())
        assert numFeatures == self.numSignals * self.numSignalFeatures, f"{numFeatures} {self.numSignals} {self.numSignalFeatures}"

        # Reshape the inputs to integrate into the model's expected format.
        signalFeatures = reshapedSignalFeatures.view((batchSize, self.numSignals, self.numSignalFeatures))

        # predict the activities.
        activityDistribution = self.forward(signalFeatures, predictActivity=True, allSignalFeatures=True)

        return activityDistribution.detach().numpy()
