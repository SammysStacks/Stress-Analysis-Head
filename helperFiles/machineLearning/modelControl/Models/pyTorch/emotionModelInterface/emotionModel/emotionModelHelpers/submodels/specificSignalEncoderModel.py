from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.waveletOperator.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class specificSignalEncoderModel(nn.Module):

    def __init__(self, sequenceLength, numInitialLayers, numFinalLayers, finalSignalDim, operatorParameters):
        super(specificSignalEncoderModel, self).__init__()
        # General model parameters.
        self.numInitialLayers = numInitialLayers  # The number of initial layers for the signal encoder.
        self.numFinalLayers = numFinalLayers  # The number of final layers for the signal encoder.
        self.finalSignalDim = finalSignalDim  # The final dimension of the signals.

        # Neural operator parameters.
        self.numDecompositions = min(5, waveletNeuralOperatorLayer.max_decompositions(signal_length=self.sequenceBounds[0], wavelet_name=operatorParameters['waveletOperator']['waveletType']))  # Number of decompositions for the waveletType transform.
        self.waveletType = operatorParameters['waveletOperator']['waveletType']  # wavelet type for the waveletType transform: bior, db3, dmey
        self.activationMethod = self.getActivationMethod()
        self.mode = 'zero'  # Mode for the waveletType transform.

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def initialLearning(self, signalData):
        pass

    def finalLearning(self, signalData):
        pass
