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

    def __init__(self, operatorType, encodedDimension, fourierDimension, numModelLayers, goldenRatio, activationMethod, learningProtocol, neuralOperatorParameters):
        super(sharedSignalEncoderModel, self).__init__(sequenceLength=fourierDimension, numInputSignals=1, numOutputSignals=1, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.encodedTimeWindow = modelConstants.timeWindows[-1]  # The time window for the encoded signal.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.fourierDimension = fourierDimension  # The dimension of the fourier signal.
        self.numModelLayers = numModelLayers  # The number of model layers to use.
        self.operatorType = operatorType  # The operator type for the neural operator.
        self.goldenRatio = goldenRatio  # The golden ratio for the model.

        # Initialize the pseudo-encoded times for the fourier data.
        pseudoEncodedTimes = torch.arange(0, self.encodedTimeWindow, step=self.encodedTimeWindow/self.encodedDimension)
        self.register_buffer(name='pseudoEncodedTimes', tensor=torch.flip(pseudoEncodedTimes, dims=[0]))  # Non-learnable parameter.

        # The neural layers for the signal encoder.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.processingLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        for layerInd in range(self.numModelLayers): self.addLayer()

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction = None
        self.testingLosses_signalReconstruction = None
        self.trainingLosses_manifoldProjection = None
        self.testingLosses_manifoldProjection = None

        # Reset the model.
        self.resetModel()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=1))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerFC(numSignals=1, sequenceLength=self.fourierDimension))
        else: raise "The learning protocol is not yet implemented."

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        self.trainingLosses_manifoldProjection = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_manifoldProjection = []  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

    def forwardFFT(self, inputData):
        # Perform the forward FFT and extract the magnitude and phase.
        fourierData = torch.fft.rfft(inputData, n=self.encodedDimension, dim=-1, norm='ortho')
        fourierMagnitudeData = fourierData.abs()
        fourierPhaseData = fourierData.angle()

        return fourierMagnitudeData, fourierPhaseData

    def backwardFFT(self, fourierMagnitudeData, fourierPhaseData):
        # Reconstruct the fourier data from the magnitude and phase.
        fourierData = fourierMagnitudeData * torch.exp(1j * fourierPhaseData)
        initialData = torch.fft.irfft(fourierData, n=self.encodedDimension, dim=-1, norm='ortho')

        return initialData

    def learningInterface(self, layerInd, signalData):
        # Reshape the signal data.
        batchSize, numSignals, signalLength = signalData.shape
        signalData = signalData.view(batchSize*numSignals, 1, signalLength)

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
