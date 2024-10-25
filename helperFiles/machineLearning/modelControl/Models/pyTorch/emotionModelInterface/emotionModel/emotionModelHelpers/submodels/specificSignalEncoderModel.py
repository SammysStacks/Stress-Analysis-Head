from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, numExperiments, operatorType, encodedDimension, fourierDimension, numSignals, numLiftingLayers, numModelLayers, goldenRatio, activationMethod, learningProtocol, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=fourierDimension, numInputSignals=numSignals*numLiftingLayers, numOutputSignals=numSignals*numLiftingLayers, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.fourierDimension = fourierDimension  # The dimension of the fourier signal.
        self.numLiftingLayers = numLiftingLayers  # The number of lifting layers to use.
        self.numModelLayers = numModelLayers  # The number of model layers to use.
        self.goldenRatio = goldenRatio  # The golden ratio for the model.
        self.numSignals = numSignals  # The number of signals to encode.

        # The neural layers for the signal encoder.
        self.processingLayers, self.neuralLayers, self.addingFlags = nn.ModuleList(), nn.ModuleList(), []
        for layerInd in range(1 + self.numModelLayers // self.goldenRatio): self.addLayer()

        # Initialize the blank signal profile.
        self.physiologicalProfileAnsatz = self.getInitialPhysiologicalProfile(numExperiments=numExperiments, encodedDimension=encodedDimension)

        # Assert the validity of the input parameters.
        assert self.numModelLayers % self.goldenRatio == 0, "The number of model layers must be divisible by the golden ratio."
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
        # Create the layers.
        self.addingFlags.append(not self.addingFlags[-1] if len(self.addingFlags) != 0 else True)
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.fourierDimension))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerRFC(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.fourierDimension))
        else: raise "The learning protocol is not yet implemented."

    def getCurrentPhysiologicalProfile(self, batchInds):
        # batchInds: The indices of the signals to estimate. Dims: batchSize
        return self.physiologicalProfileAnsatz[batchInds]

    def learningInterface(self, layerInd, signalData):
        # For the forward/harder direction.
        if reversibleInterface.forwardDirection:
            # Apply the neural operator layer with activation.
            signalData = self.neuralLayers[layerInd](signalData)
            signalData = self.activationFunction(signalData, addingFlag=self.addingFlags[layerInd])

            # Apply the post-processing layer.
            signalData = self.processingLayers[layerInd](signalData)
        else:
            # Get the reverse layer index.
            pseudoLayerInd = len(self.neuralLayers) - layerInd - 1
            assert 0 <= pseudoLayerInd < len(self.neuralLayers), f"The pseudo layer index is out of bounds: {pseudoLayerInd}, {len(self.neuralLayers)}, {layerInd}"

            # Apply the neural operator layer with activation.
            signalData = self.processingLayers[pseudoLayerInd](signalData)

            # Apply the neural operator layer with activation.
            signalData = self.activationFunction(signalData, addingFlag=self.addingFlags[pseudoLayerInd])
            signalData = self.neuralLayers[pseudoLayerInd](signalData)

        return signalData
