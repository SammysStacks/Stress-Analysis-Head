# General

import matplotlib.pyplot as plt
import torch
from torch import nn

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
# Import files for machine learning
from .modelComponents.transformerHelpers.attentionMethods import attentionMethods
from ..generalMethods.generalMethods import generalMethods
from ..modelConstants import modelConstants


class sharedSignalEncoderModel(nn.Module):

    def __init__(self, numAttentionLayers, latentQueryKeyDim, finalSignalDim, numEncodedSignals, encodedSamplingFreq, numSigEncodingLayers, numSigLiftedChannels, waveletType, debuggingResults=False):
        super(sharedSignalEncoderModel, self).__init__()
        # General model parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Signal encoder parameters.
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = numSigEncodingLayers  # The number of operator layers during signal encoding.
        self.encodedSamplingFreq = encodedSamplingFreq    # The sampling frequency of the encoded signals.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.waveletType = waveletType  # The type to use during the signal encoder.

        self.attentionMechanisms = nn.ModuleList()
        for layer in range(numAttentionLayers):
            # Initialize the signal encoder modules.
            self.attentionMechanisms.append(attentionMethods(inputQueryKeyDim=1, latentQueryKeyDim=latentQueryKeyDim, inputValueDim=1, latentValueDim=finalSignalDim, numHeads=1, addBias=False))

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
        # For each layer, apply the attention mechanism.
        for layerInd in range(len(self.attentionMechanisms)):
            signalData = self.attentionMechanisms[layerInd](signalData)

        return signalData

    def reconstructEncodedData(self, encodedData, numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False):
        # reconstructedInitEncodingData dimension: batchSize, numSignals, finalDistributionLength
        denoisedReconstructedData = self.encodeSignals.denoiseSignals.applyDenoiser()

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

    @staticmethod
    def plotDataFlowDetails(initialSignalData, positionEncodedData, encodedData, decodedData, reconstructedData, denoisedReconstructedData):
        fig = plt.figure()
        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2, label="Initial Data")
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.title("Positional Encoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.plot(encodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2, label="Encoded Data")
        plt.title("Signal Encoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(encodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2, label="Encoded Data")
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.plot(decodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, alpha=0.5, label="Decoded Data (Backward Path)")
        plt.title("Position Decoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(decodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, alpha=0.5, label="Decoded Data (Backward Path)")
        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2, label="Initial Data")
        plt.plot(reconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.5, label="Reconstructed Data")
        plt.plot(denoisedReconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.25, label="Denoised Reconstructed Data")
        plt.title("Final Denoising"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)
