# General

import torch
from torch import nn

from .modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from .modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
# Import files for machine learning
from ..generalMethods.generalMethods import generalMethods
from ..optimizerMethods import activationFunctions


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, neuralOperatorParameters, numOperatorLayers, activationMethod):
        super(sharedSignalEncoderModel, self).__init__(sequenceLength=encodedDimension, numInputSignals=1, numOutputSignals=1, addBiasTerm=False)
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
            self.processingLayers.append(self.postProcessingLayer(inChannel=1))

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction = None
        self.testingLosses_signalReconstruction = None
        self.trainingLosses_manifoldProjection = None
        self.testingLosses_manifoldProjection = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        self.trainingLosses_manifoldProjection = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_manifoldProjection = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

    def sharedLearning(self, signalData):
        # Reshape the signal data.
        batchSize, numSignals, signalLength = signalData.shape
        signalData = signalData.view(batchSize*numSignals, 1, signalLength)

        for layerInd in range(self.numOperatorLayers):
            if reversibleInterface.forwardDirection:
                # Apply the neural operator layer with activation.
                signalData = self.neuralLayers[layerInd](signalData)
                signalData = self.activationFunction(signalData, layerInd % 2 == 0)

                # Apply the post-processing layer.
                signalData = self.processingLayers[layerInd](signalData)
            else:
                # Apply the post-processing layer.
                pseudoLayerInd = self.numOperatorLayers - layerInd - 1
                signalData = self.processingLayers[pseudoLayerInd](signalData)

                # Apply the neural operator layer with activation.
                signalData = self.activationFunction(signalData, pseudoLayerInd % 2 == 0)
                signalData = self.neuralLayers[pseudoLayerInd](signalData)

        # Reshape the signal data.
        signalData = signalData.view(batchSize, numSignals, signalLength)

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
