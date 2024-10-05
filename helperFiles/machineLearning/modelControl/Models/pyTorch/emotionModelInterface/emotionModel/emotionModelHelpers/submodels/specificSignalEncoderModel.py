# General

from torch import nn

# Import files for machine learning


class specificSignalEncoderModel(nn.Module):

    def __init__(self, latentQueryKeyDim, finalSignalDim):
        super(specificSignalEncoderModel, self).__init__()
        # General model parameters.
        self.latentQueryKeyDim = latentQueryKeyDim  # The embedded dimension of the query and keys: Int
        self.finalSignalDim = finalSignalDim  # The final dimension of the signals.

        # Method to converge to a final signal length (now pseudo-evenly sampled).
        self.liftingOperator = nn.Linear(finalSignalDim, finalSignalDim, bias=True)

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def projectionOperator(self, signalData):
        return self.liftingOperator(signalData)

    def initialLearning(self, signalData):
        pass

    def finalLearning(self, signalData):
        pass
