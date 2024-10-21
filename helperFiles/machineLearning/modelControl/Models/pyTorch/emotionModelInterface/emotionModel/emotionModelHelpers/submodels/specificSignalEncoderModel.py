import math

import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, numExperiments, operatorType, encodedDimension, fourierDimension, numOperatorLayers, numSignals, numLiftingLayers, activationMethod, learningProtocol, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(sequenceLength=fourierDimension, numInputSignals=numSignals*numLiftingLayers, numOutputSignals=numSignals, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.numOperatorLayers = numOperatorLayers  # The number of operator layers to use.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.fourierDimension = fourierDimension  # The dimension of the fourier signal.
        self.numLiftingLayers = numLiftingLayers  # The number of lifting layers to use.
        self.operatorType = operatorType  # The operator type for the neural operator.

        # The neural layers for the signal encoder.
        self.initialProcessingLayers = nn.ModuleList()
        self.finalProcessingLayers = nn.ModuleList()
        self.initialNeuralLayers = nn.ModuleList()
        self.finalNeuralLayers = nn.ModuleList()

        for layerInd in range(self.numOperatorLayers):
            # Create the initial layers.
            self.initialNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            if self.learningProtocol == 'rCNN': self.initialProcessingLayers.append(self.postProcessingLayerCNN(numSignals=numSignals*numLiftingLayers))
            elif self.learningProtocol == 'rFC': self.initialProcessingLayers.append(self.postProcessingLayerFC(numSignals=numSignals*numLiftingLayers, sequenceLength=fourierDimension))
            else: raise "The learning protocol is not yet implemented."

            # Create the final layers.
            self.finalNeuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            if self.learningProtocol == 'rCNN': self.finalProcessingLayers.append(self.postProcessingLayerCNN(numSignals=numSignals*numLiftingLayers))
            elif self.learningProtocol == 'rFC': self.finalProcessingLayers.append(self.postProcessingLayerFC(numSignals=numSignals*numLiftingLayers, sequenceLength=fourierDimension))
            else: raise "The learning protocol is not yet implemented."

        # Initialize the blank signal profile.
        physiologicalProfileAnsatz = nn.Parameter(torch.randn(numExperiments, encodedDimension, dtype=torch.float64))
        self.physiologicalProfileAnsatz = nn.init.normal_(physiologicalProfileAnsatz, mean=0, std=0.25)

        # Assert the validity of the input parameters.
        assert 0 < self.numOperatorLayers, "The number of operator layers must be greater than 0."
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def getPhysiologicalProfileEstimation(self, batchInds, trainingFlag):
        # batchInds: The indices of the signals to estimate. Dims: batchSize
        if trainingFlag: return self.physiologicalProfileAnsatz[batchInds]
        batchSize = batchInds.size()

        # Initialize the blank signal profile.
        physiologicalProfileGuess = nn.Parameter(torch.randn(batchSize, self.encodedDimension, dtype=torch.float64))
        return nn.init.kaiming_uniform_(physiologicalProfileGuess, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

    def signalSpecificInterface(self, signalData, initialModel):
        if initialModel: return self.initialLearning(signalData, self.initialNeuralLayers, self.initialProcessingLayers)
        else: return self.initialLearning(signalData, self.finalNeuralLayers, self.finalProcessingLayers)

    def initialLearning(self, signalData, neuralLayers, processingLayers):
        # For each initial layer.
        for layerInd in range(self.numOperatorLayers):
            if reversibleInterface.forwardDirection:
                # Apply the neural operator layer with activation.
                signalData = neuralLayers[layerInd](signalData)
                signalData = self.activationFunction(signalData, layerInd % 2 == 0)

                # Apply the post-processing layer.
                signalData = processingLayers[layerInd](signalData)
            else:
                # Apply the post-processing layer.
                pseudoLayerInd = self.numOperatorLayers - layerInd - 1
                signalData = processingLayers[pseudoLayerInd](signalData)

                # Apply the neural operator layer with activation.
                signalData = self.activationFunction(signalData, pseudoLayerInd % 2 == 0)
                signalData = neuralLayers[pseudoLayerInd](signalData)

        return signalData.contiguous()
