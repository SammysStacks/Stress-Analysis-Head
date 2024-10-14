# General

import matplotlib.pyplot as plt
import torch
from torch import nn

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from .modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from .modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
# Import files for machine learning
from .modelComponents.transformerHelpers.attentionMethods import attentionMethods
from ..generalMethods.generalMethods import generalMethods
from ..modelConstants import modelConstants
from ..optimizerMethods import activationFunctions


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, latentQueryKeyDim, neuralOperatorParameters, sequenceLength, numOperatorLayers, numInputSignals, activationMethod):
        super(sharedSignalEncoderModel, self).__init__(sequenceLength=sequenceLength, numInputSignals=numInputSignals, numOutputSignals=numInputSignals, addBiasTerm=False)
        # General model parameters.
        self.numOperatorLayers = numOperatorLayers  # The number of operator layers to use.
        self.operatorType = operatorType  # The type of operator to use for the neural operator.

        # The neural layers for the signal encoder.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.processingLayers = nn.ModuleList()
        self.neuralLayers = nn.ModuleList()

        # Create the operator layers.
        for layerInd in range(self.numOperatorLayers):
            self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            self.processingLayers.append(self.postProcessingLayer(inChannel=numInputSignals, groups=numInputSignals))

        # Initialize the signal encoder modules.
        self.attentionMechanism = attentionMethods(inputQueryKeyDim=1, latentQueryKeyDim=latentQueryKeyDim, inputValueDim=1, latentValueDim=encodedDimension, numHeads=1, addBias=False)

        # Initialize loss holders.
        self.trainingLosses_timeReconstructionAnalysis = None
        self.testingLosses_timeReconstructionAnalysis = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_timeReconstructionAnalysis = [[] for _ in modelConstants.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionAnalysis = [[] for _ in modelConstants.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

    def learnedInterpolation(self, signalData):
        """ signalData: batchSize, numSignals, signalSpecificLength* """
        interpolatedSignalData = self.attentionMechanism(signalData)
        # interpolatedSignalData: batchSize, numSignals, encodedDimension

        return interpolatedSignalData

    def sharedLearning(self, signalData):
        for layerInd in range(self.numOperatorLayers):
            if reversibleInterface.forwardDirection:
                # Apply the neural operator layer with activation.
                signalData = self.neuralLayers[layerInd](signalData)
                signalData = self.activationFunction(signalData)

                # Apply the post-processing layer.
                signalData = self.processingLayers[layerInd](signalData)
            else:
                # Apply the post-processing layer.
                layerInd = self.numOperatorLayers - layerInd - 1
                signalData = self.processingLayers[layerInd](signalData)

                # Apply the neural operator layer with activation.
                signalData = self.activationFunction(signalData)
                signalData = self.neuralLayers[layerInd](signalData)

        return signalData

    def calculateOptimalLoss(self, initialSignalData, printLoss=True):
        with torch.no_grad():
            # Perform the optimal compression via PCA and embed channel information (for reconstruction).
            pcaProjection, principal_components = generalMethods.svdCompression(initialSignalData, self.numEncodedSignals, standardizeSignals=True)
            # Loss for PCA reconstruction
            pcaReconstruction = torch.matmul(principal_components, pcaProjection)
            pcaReconstruction = (pcaReconstruction + initialSignalData.mean(dim=-1, keepdim=True)) * initialSignalData.std(dim=-1, keepdim=True)
            pcaReconstructionLoss = (initialSignalData - pcaReconstruction).pow(2).mean(dim=2).mean(dim=1)
            if printLoss: print("\tFIRST Optimal Compression Loss STD:", pcaReconstructionLoss.mean().item())

            return pcaReconstructionLoss
