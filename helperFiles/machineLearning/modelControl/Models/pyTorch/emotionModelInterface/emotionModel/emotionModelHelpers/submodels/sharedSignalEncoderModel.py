from matplotlib import pyplot as plt
from torch import nn
import torch
import math

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.optimizerMethods import activationFunctions
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, numModelLayers, goldenRatio, activationMethod, learningProtocol, neuralOperatorParameters):
        super(sharedSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numInputSignals=1, numOutputSignals=1, learningProtocol=learningProtocol, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.encodedTimeWindow = modelConstants.timeWindows[-1]  # The time window for the encoded signal.
        self.fourierDimension = encodedDimension // 2 + 1  # The dimension of the fourier data.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numModelLayers = numModelLayers  # The number of model layers to use.
        self.goldenRatio = goldenRatio  # The golden ratio for the model.

        # Initialize the pseudo-encoded times for the fourier data.
        pseudoEncodedTimes = torch.linspace(start=0, end=self.encodedTimeWindow, steps=self.encodedDimension).flip(dims=[0])
        self.register_buffer(name='pseudoEncodedTimes', tensor=pseudoEncodedTimes)  # Non-learnable parameter.
        deltaTimes = torch.unique(self.pseudoEncodedTimes.diff().round(decimals=4))
        assert len(deltaTimes) == 1, f"The time gaps are not similar: {deltaTimes}"

        # Initialize the frequency indices for the fourier data.
        self.imagAngularFrequencies = 2j * torch.pi * torch.fft.rfftfreq(self.encodedDimension, d=self.encodedTimeWindow / (self.encodedDimension - 1), dtype=torch.float64)
        self.imagAngularFrequencies = self.imagAngularFrequencies.view(1, 1, 1, self.fourierDimension)
        # angularFrequencies: 1, 1, 1, fourierDimension

        # The neural layers for the signal encoder.
        self.activationFunction = activationFunctions.getActivationMethod(activationMethod=activationMethod)
        self.processingLayers, self.neuralLayers, self.addingFlags = nn.ModuleList(), nn.ModuleList(), []
        for layerInd in range(self.numModelLayers): self.addLayer()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        # Create the layers.
        self.addingFlags.append(not self.addingFlags[-1] if len(self.addingFlags) != 0 else True)
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=1, sequenceLength=self.encodedDimension))
        elif self.learningProtocol == 'rFC': self.processingLayers.append(self.postProcessingLayerRFC(numSignals=1, sequenceLength=self.encodedDimension))
        else: raise "The learning protocol is not yet implemented."

        # Adjust the addingFlag to account for the specific layers.
        if len(self.addingFlags) % self.goldenRatio == 1: self.addingFlags[-1] = not self.addingFlags[-1]

    def forwardFFT(self, inputData):
        # Perform the forward FFT and extract the magnitude and phase.
        fourierData = torch.fft.rfft(inputData, n=self.encodedDimension, dim=-1, norm='ortho')
        imaginaryFourierData = fourierData.imag
        realFourierData = fourierData.real

        return realFourierData, imaginaryFourierData

    def backwardFFT(self, realFourierData, imaginaryFourierData, resampledTimes=None):
        # Reconstruct the fourier data from the real and imaginary components.
        fourierData = realFourierData + 1j * imaginaryFourierData
        # fourierData: batchSize, numSignals, fourierDimension

        # Reconstruct the data based on the physiological times.
        if resampledTimes is None: return torch.fft.irfft(fourierData, n=self.encodedDimension, dim=-1, norm='ortho')

        # Prepare the information for resampling.
        batchSize, numSignals, maxSequenceLength = resampledTimes.size()
        self.imagAngularFrequencies = self.imagAngularFrequencies.to(realFourierData.device)
        fullFourierData = fourierData.view(batchSize, numSignals, 1, self.fourierDimension)
        # angularFrequencies: 1, 1, 1, fourierDimension

        # Reconstruct the data based on the new sampled times.
        basisFunctions = torch.exp(self.imagAngularFrequencies * resampledTimes.flip(dims=[-1]).view(batchSize, numSignals, maxSequenceLength, 1))
        reconstructedData = torch.sum(fullFourierData * basisFunctions, dim=-1) * 2 / math.sqrt(self.encodedDimension)
        # basisFunctions: batchSize, numSignals, maxSequenceLength, fourierDimension
        # fullFourierData: batchSize, numSignals, 1, fourierDimension
        # reconstructedData: batchSize, numSignals, maxSequenceLength

        return reconstructedData.real

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


if __name__ == "__main__":
    # General parameters.
    torch.random.manual_seed(0)
    _batchSize, _numSignals, _encodedDimension = 2, 3, 1024
    _sequenceLength = 32

    # Initialize the shared signal encoder model.
    sharedSignalEncoderModelClass = sharedSignalEncoderModel(operatorType='wavelet', encodedDimension=_encodedDimension, numModelLayers=1, goldenRatio=16, activationMethod='nonLinearAddition',
                                                             learningProtocol='rCNN', neuralOperatorParameters={'wavelet': {}})
    encodedTimeWindow = sharedSignalEncoderModelClass.encodedTimeWindow
    _signalTimes = sharedSignalEncoderModelClass.pseudoEncodedTimes

    # Initialize the signal data (e.g., a sine wave for testing).
    _signalData = torch.sin(2 * torch.pi * 2 * _signalTimes) + torch.sin(2 * torch.pi * 0.01 * _signalTimes) + torch.sin(2 * torch.pi * 9 * _signalTimes)
    _signalData = 10*_signalData.unsqueeze(0).unsqueeze(0).expand(_batchSize, _numSignals, _encodedDimension) + torch.randn(_batchSize, _numSignals, _encodedDimension)*5
    _sampledTimes = torch.linspace(start=0, end=encodedTimeWindow, steps=_sequenceLength).unsqueeze(0).unsqueeze(0).expand(_batchSize, _numSignals, _sequenceLength).flip(dims=[-1])

    # Cast to a double.
    _sampledTimes = _sampledTimes.double()
    _signalData = _signalData.double()

    # Compute the forward FFT.
    _realFourierData, _imaginaryFourierData = sharedSignalEncoderModelClass.forwardFFT(_signalData)

    # Compute the backward FFT.
    _reconstructedData = sharedSignalEncoderModelClass.backwardFFT(_realFourierData, _imaginaryFourierData, resampledTimes=_sampledTimes)
    _resampledData = sharedSignalEncoderModelClass.backwardFFT(_realFourierData, _imaginaryFourierData, resampledTimes=None)

    # Plot the results.
    plt.plot(_signalTimes.numpy(), _signalData[0][0].numpy(), 'k-o', label='Original', markersize=2)
    plt.plot(_sampledTimes[0][0].numpy(), _reconstructedData[0][0].real.numpy(), 'o-', color='tab:green', label='real', markersize=3)
    plt.plot(_signalTimes.numpy(), _resampledData[0][0].numpy(), 'o-', color='tab:blue', label='Reconstructed', markersize=2)
    plt.xlim(0, 30)
    plt.legend()
    plt.show()
