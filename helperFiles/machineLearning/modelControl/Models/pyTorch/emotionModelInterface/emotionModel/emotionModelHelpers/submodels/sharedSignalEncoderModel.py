# General

import torch
from torch import nn

from .modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from .modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
# Import files for machine learning
from ..generalMethods.generalMethods import generalMethods
from ..modelConstants import modelConstants
from ..optimizerMethods import activationFunctions


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, fourierDimension, numOperatorLayers, activationMethod, learningProtocol, neuralOperatorParameters):
        super(sharedSignalEncoderModel, self).__init__(sequenceLength=fourierDimension, numInputSignals=1, numOutputSignals=1, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.encodedTimeWindow = modelConstants.timeWindows[-1]  # The time window for the encoded signal.
        self.numOperatorLayers = numOperatorLayers  # The number of operator layers to use.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.fourierDimension = fourierDimension  # The dimension of the fourier signal.
        self.operatorType = operatorType  # The operator type for the neural operator.

        # Initialize the pseudo-encoded times for the fourier data.
        pseudoEncodedTimes = torch.arange(0, self.encodedTimeWindow, step=self.encodedTimeWindow/self.encodedDimension)
        self.register_buffer(name='pseudoEncodedTimes', tensor=torch.flip(pseudoEncodedTimes, dims=[0]))  # Non-learnable parameter.

        # The neural layers for the signal encoder.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.processingLayers = nn.ModuleList()
        self.neuralLayers = nn.ModuleList()

        # Create the operator layers.
        for layerInd in range(self.numOperatorLayers):
            self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=neuralOperatorParameters))
            if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=1))
            elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerFC(numSignals=1, sequenceLength=fourierDimension))
            else: raise "The learning protocol is not yet implemented."

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

    def forwardFFT(self, inputData):
        return torch.fft.rfft(inputData, n=self.encodedDimension, dim=-1, norm='ortho')

    def backwardFFT(self, inputData):
        return torch.fft.irfft(inputData, n=self.encodedDimension, dim=-1, norm='ortho')

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
