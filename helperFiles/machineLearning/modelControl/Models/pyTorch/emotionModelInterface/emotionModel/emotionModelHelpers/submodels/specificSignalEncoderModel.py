from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.trainingProfileInformation import trainingProfileInformation


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, numExperiments, operatorType, encodedDimension, numSignals, numLiftingLayers, goldenRatio, learningProtocol, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numInputSignals=numSignals*numLiftingLayers, numOutputSignals=numSignals*numLiftingLayers, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.activationMethod = self.getActivationType()  # The activation method to use.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numLiftingLayers = numLiftingLayers  # The number of lifting layers to use.
        self.goldenRatio = goldenRatio  # The golden ratio for the model.
        self.numSignals = numSignals  # The number of signals to encode.

        # The neural layers for the signal encoder.
        self.processingLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        self.profileModel = trainingProfileInformation(numExperiments, encodedDimension)

        # Assert the validity of the input parameters.
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction = None
        self.testingLosses_signalReconstruction = None
        self.resetModel()

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: numEpochs

    def addLayer(self):
        switchActivationDirection = True
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True, switchActivationDirection=switchActivationDirection))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.encodedDimension, activationMethod=self.activationMethod, switchActivationDirection=not switchActivationDirection))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerRFC(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.encodedDimension, activationMethod=self.activationMethod, switchActivationDirection=not switchActivationDirection))
        else: raise "The learning protocol is not yet implemented."

    def learningInterface(self, layerInd, signalData):
        # For the forward/harder direction.
        if not reversibleInterface.forwardDirection:
            # Apply the neural operator layer with activation.
            signalData = self.neuralLayers[layerInd].reversibleInterface(signalData)
            signalData = self.processingLayers[layerInd](signalData)
        else:
            # Get the reverse layer index.
            pseudoLayerInd = len(self.neuralLayers) - layerInd - 1
            assert 0 <= pseudoLayerInd < len(self.neuralLayers), f"The pseudo layer index is out of bounds: {pseudoLayerInd}, {len(self.neuralLayers)}, {layerInd}"

            # Apply the neural operator layer with activation.
            signalData = self.processingLayers[pseudoLayerInd](signalData)
            signalData = self.neuralLayers[pseudoLayerInd].reversibleInterface(signalData)

        return signalData.contiguous()

    def printParams(self):
        # Count the trainable parameters.
        numParams = (sum(p.numel() for p in self.parameters() if p.requires_grad) - self.profileModel.physiologicalProfile.size(0) * self.encodedDimension) / self.numSignals
        print(f'The model has {numParams} trainable parameters per signal; {numParams*self.numSignals} total parameters.')


if __name__ == "__main__":
    # General parameters.
    _neuralOperatorParameters = modelParameters.getNeuralParameters({'waveletType': 'bior3.7'})['neuralOperatorParameters']
    _batchSize, _numSignals, _sequenceLength = 2, 88, 256

    # Set up the parameters.
    neuralLayerClass = specificSignalEncoderModel(numExperiments=_batchSize, operatorType='wavelet', encodedDimension=_sequenceLength, numSignals=_numSignals, numLiftingLayers=1, goldenRatio=4, learningProtocol='rCNN', neuralOperatorParameters=_neuralOperatorParameters)
    neuralLayerClass.addLayer()

    # Print the number of trainable parameters.
    neuralLayerClass.printParams()
