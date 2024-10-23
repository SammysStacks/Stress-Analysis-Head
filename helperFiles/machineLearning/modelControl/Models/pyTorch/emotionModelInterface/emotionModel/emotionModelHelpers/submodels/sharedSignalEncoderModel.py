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
        pseudoEncodedTimes = torch.linspace(start=0, end=self.encodedTimeWindow, steps=self.encodedDimension).flip(dims=[0])
        self.register_buffer(name='pseudoEncodedTimes', tensor=pseudoEncodedTimes)  # Non-learnable parameter.

        # The neural layers for the signal encoder.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.processingLayers, self.neuralLayers, self.addingFlags = nn.ModuleList(), nn.ModuleList(), []
        for layerInd in range(self.numModelLayers): self.addLayer()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.addingFlags.append(not self.addingFlags[-1] if len(self.addingFlags) != 0 else True)
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=1))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerRFC(numSignals=1, sequenceLength=self.fourierDimension))
        else: raise "The learning protocol is not yet implemented."

        # Adjust the addingFlag to account for the specific layers.
        if len(self.addingFlags) % self.goldenRatio == 1: self.addingFlags[-1] = not self.addingFlags[-1]

    def forwardFFT(self, inputData):
        # Perform the forward FFT and extract the magnitude and phase.
        fourierData = torch.fft.rfft(inputData, n=self.encodedDimension, dim=-1, norm='ortho')
        imaginaryFourierData = fourierData.imag
        realFourierData = fourierData.real

        return realFourierData, imaginaryFourierData

    def backwardFFT(self, realFourierData, imaginaryFourierData):
        # Reconstruct the fourier data and the initial data.
        fourierData = realFourierData + 1j * imaginaryFourierData
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
