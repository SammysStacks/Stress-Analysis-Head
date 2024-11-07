from torch import nn

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
        self.trainingLosses_smoothPhysiology = None
        self.testingLosses_smoothPhysiology = None
        self.trainingLosses_smoothResampled = None
        self.testingLosses_smoothResampled = None
        self.resetModel()

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: numEpochs

        # Signal encoder physiological profile loss holders.
        self.trainingLosses_smoothPhysiology = []  # List of list of physiological profile training losses. Dim: numEpochs
        self.testingLosses_smoothPhysiology = []  # List of list of physiological profile testing losses. Dim: numEpochs

        # Signal encoder resampled signal loss holders.
        self.trainingLosses_smoothResampled = []  # List of list of resampled signal training losses. Dim: numEpochs
        self.testingLosses_smoothResampled = []  # List of list of resampled signal testing losses. Dim: numEpochs

    def addLayer(self):
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True, switchActivationDirection=True))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.encodedDimension, activationMethod=self.activationMethod, switchActivationDirection=False))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerRFC(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.encodedDimension, activationMethod=self.activationMethod, switchActivationDirection=False))
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
