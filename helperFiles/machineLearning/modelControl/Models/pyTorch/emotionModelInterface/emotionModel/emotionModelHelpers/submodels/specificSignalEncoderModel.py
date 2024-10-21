import math

import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, numExperiments, operatorType, encodedDimension, fourierDimension, numSignals, numLiftingLayers, numModelLayers, goldenRatio, activationMethod, learningProtocol, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(sequenceLength=fourierDimension, numInputSignals=numSignals*numLiftingLayers, numOutputSignals=numSignals, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.fourierDimension = fourierDimension  # The dimension of the fourier signal.
        self.numLiftingLayers = numLiftingLayers  # The number of lifting layers to use.
        self.numModelLayers = numModelLayers  # The number of model layers to use.
        self.operatorType = operatorType  # The operator type for the neural operator.
        self.goldenRatio = goldenRatio  # The golden ratio for the model.
        self.numSignals = numSignals  # The number of signals to encode.

        # The neural layers for the signal encoder.
        self.processingLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        for layerInd in range(1 + self.numModelLayers // self.goldenRatio): self.addLayer()

        # Initialize the blank signal profile.
        self.physiologicalProfileAnsatz = nn.Parameter(torch.randn(numExperiments, encodedDimension, dtype=torch.float64))
        self.physiologicalProfileAnsatz = nn.init.normal_(self.physiologicalProfileAnsatz, mean=0, std=0.25)

        # Assert the validity of the input parameters.
        assert self.numModelLayers % self.goldenRatio == 0, "The number of model layers must be divisible by the golden ratio."
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=self.numSignals*self.numLiftingLayers))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerFC(numSignals=self.numSignals*self.numLiftingLayers, sequenceLength=self.fourierDimension))
        else: raise "The learning protocol is not yet implemented."

    def getPhysiologicalProfileEstimation(self, batchInds, trainingFlag):
        # batchInds: The indices of the signals to estimate. Dims: batchSize
        if trainingFlag: return self.physiologicalProfileAnsatz[batchInds]
        batchSize = batchInds.size(0)

        # Initialize the blank signal profile.
        physiologicalProfileGuess = nn.Parameter(torch.randn(size=(batchSize, self.encodedDimension), dtype=torch.float64, device=batchInds.device))
        return nn.init.kaiming_uniform_(physiologicalProfileGuess, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

    def learningInterface(self, layerInd, signalData):
        # For the forward/harder direction.
        if reversibleInterface.forwardDirection:
            # Apply the neural operator layer with activation.
            signalData = self.neuralLayers[layerInd](signalData)
            signalData = self.activationFunction(signalData, layerInd % 2 == 0)

            # Apply the post-processing layer.
            signalData = self.processingLayers[layerInd](signalData)
        else:
            # Get the reverse layer index.
            pseudoLayerInd = len(self.neuralLayers) - layerInd - 1
            assert 0 <= pseudoLayerInd < len(self.neuralLayers), f"The pseudo layer index is out of bounds: {pseudoLayerInd}, {len(self.neuralLayers)}, {layerInd}"

            # Apply the neural operator layer with activation.
            signalData = self.processingLayers[pseudoLayerInd](signalData)

            # Apply the neural operator layer with activation.
            signalData = self.activationFunction(signalData, pseudoLayerInd % 2 == 0)
            signalData = self.neuralLayers[pseudoLayerInd](signalData)

        return signalData
