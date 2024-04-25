# PyTorch
import torch

# Import files for machine learning
from .modelComponents.generalSignalEncoder import generalSignalEncoding  # Framework for encoding/decoding of all signals.
from .helperModules.trainingSignalEncoder import trainingSignalEncoder
from ..generalMethods.generalMethods import generalMethods
from ...._globalPytorchModel import globalModel


class signalEncoderModel(globalModel):
    def __init__(self, sequenceBounds, maxNumSignals, numEncodedSignals, numExpandedSignals, numEncodingLayers, numLiftedChannels, timeWindows, accelerator):
        super(signalEncoderModel, self).__init__()
        # General model parameters.
        self.numExpandedSignals = numExpandedSignals  # The number of signals in the expanded form for encoding to numExpandedSignals - 1.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.numEncodingLayers = numEncodingLayers  # The number of transformer layers during signal encoding.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence lengths to consider.
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider.
        self.timeWindows = timeWindows  # A list of all time windows to consider for the encoding.
        self.accelerator = accelerator  # Hugging face interface for model and data optimizations.
        self.numLiftedChannels = numLiftedChannels
        self.numAccumulations = 0
        self.accumulatedLoss = 0

        # Method to converge to the final number of signals.
        self.encodeSignals = generalSignalEncoding(
            numExpandedSignals=self.numExpandedSignals,
            numEncodingLayers=self.numEncodingLayers,
            numLiftedChannels=self.numLiftedChannels,
            sequenceBounds=self.sequenceBounds,
            accelerator=self.accelerator,
        )

        # Initialize helper classes.
        self.trainingMethods = trainingSignalEncoder(numEncodedSignals, self.encodeSignals.expansionFactor)

        # Initialize loss holders.
        self.trainingLosses_timeReconstructionSVDAnalysis = None
        self.testingLosses_timeReconstructionSVDAnalysis = None
        self.trainingLosses_timeReconstructionAnalysis = None
        self.testingLosses_timeReconstructionAnalysis = None
        self.numEncodingsBufferPath_timeAnalysis = None
        self.trainingLosses_timeLayerAnalysis = None
        self.testingLosses_timeLayerAnalysis = None
        self.trainingLosses_timeMeanAnalysis = None
        self.testingLosses_timeMeanAnalysis = None
        self.trainingLosses_timeSTDAnalysis = None
        self.testingLosses_timeSTDAnalysis = None
        self.numEncodingsPath_timeAnalysis = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        # Time analysis loss methods.
        self.trainingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Signal encoder mean loss holders.
        self.trainingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean testing losses. Dim: numTimeWindows, numEpochs
        # Signal encoder standard deviation loss holders.
        self.trainingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation testing losses. Dim: numTimeWindows, numEpochs

        # Compression analysis.
        self.numEncodingsBufferPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of buffers at each epoch. Dim: numTimeWindows, numEpochs
        self.numEncodingsPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of the number of compressions at each epoch. Dim: numTimeWindows, numEpochs

        # Signal encoder SVF reconstruction loss holders
        self.trainingLosses_timeReconstructionSVDAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionSVDAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Keep track of gradient accumulation.
        self.numAccumulations = 0
        self.accumulatedLoss = 0

    def forward(self, signalData, initialSignalData, decodeSignals=False, calculateLoss=False, trainingFlag=False):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        print("\nEntering signal encoder model")

        # ----------------------- Data Preprocessing ----------------------- #  

        # Prepare the data for compression/expansion
        batchSize, numSignals, sequenceLength = signalData.size()
        # signalData dimension: batchSize, numSignals, sequenceLength

        # Create placeholders for the final variables.
        denoisedReconstructedData = torch.zeros_like(signalData, device=signalData.device)
        reconstructedData = torch.zeros_like(signalData, device=signalData.device)
        signalEncodingLoss = torch.zeros((batchSize,), device=signalData.device)
        # denoisedReconstructedData dimension: batchSize, numSignals, sequenceLength
        # reconstructedData dimension: batchSize, numSignals, sequenceLength
        # signalEncodingLoss dimension: batchSize

        # Initialize training parameters
        initialDecodedData = None
        decodedData = None

        # ---------------------- Training Augmentation --------------------- #  

        # Initialize augmentation parameters
        numEncodedSignals = self.numEncodedSignals
        forwardDirection = None
        totalNumEncodings = 0

        if trainingFlag:
            # Set up the training parameters
            assert decodeSignals and calculateLoss, f"Training requires decoding and loss calculations. decodeSignals: {decodeSignals}, calculateLoss: {calculateLoss}"
            numEncodedSignals, totalNumEncodings, forwardDirection = self.trainingMethods.augmentFinalTarget(numSignals)

        # ------------------- Learned Signal Compression ------------------- #

        # Learn how to add positional encoding to each signal's position.
        positionEncodedData = self.encodeSignals.positionalEncodingInterface.addPositionalEncoding(signalData)
        # positionEncodedData dimension: batchSize, numSignals, sequenceLength

        # Compress the signal space into numEncodedSignals.
        initialEncodedData, numSignalForwardPath, signalEncodingLayerLoss = self.encodeSignals(signalData=positionEncodedData, targetNumSignals=numEncodedSignals, signalEncodingLayerLoss=None,
                                                                                               calculateLoss=calculateLoss, trainingFlag=trainingFlag)
        # initialEncodedData dimension: batchSize, numEncodedSignals, sequenceLength

        # Allow the model to adjust the incoming signals
        encodedData = self.encodeSignals.finalVarianceInterface.adjustSignalVariance(initialEncodedData)
        # adjustedData dimension: batchSize, numEncodedSignals, sequenceLength

        # ---------------------- Signal Reconstruction --------------------- #
        print("Signal Encoding Downward Path:", numSignals, numSignalForwardPath, numEncodedSignals)

        if decodeSignals:
            # Perform the reverse operation.
            initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss = \
                self.reconstructEncodedData(encodedData, numSignalForwardPath, signalEncodingLayerLoss=signalEncodingLayerLoss, calculateLoss=calculateLoss, trainingFlag=trainingFlag)

        # ------------------------ Loss Calculations ----------------------- #

        if calculateLoss and decodeSignals:
            # Prepare for loss calculations.
            noisyPositionEncodedData = self.encodeSignals.dataInterface.addNoise(positionEncodedData, trainingFlag, noiseSTD=0.001)
            removedStampEncoding = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(noisyPositionEncodedData)
            # Prepare for loss calculations.
            potentialEncodedData = self.encodeSignals.finalVarianceInterface.adjustSignalVariance(signalData)
            noisyPotentialEncodedData = self.encodeSignals.dataInterface.addNoise(potentialEncodedData, trainingFlag, noiseSTD=0.001)
            potentialSignalData = self.encodeSignals.finalVarianceInterface.unAdjustSignalVariance(noisyPotentialEncodedData)

            # Calculate the loss by comparing encoder/decoder outputs.
            varReconstructionStateLoss = (initialEncodedData - initialDecodedData).pow(2).mean(dim=2).mean(dim=1)
            encodingReconstructionStateLoss = (positionEncodedData - decodedData).pow(2).mean(dim=2).mean(dim=1)
            finalReconstructionStateLoss = (signalData - reconstructedData).pow(2).mean(dim=2).mean(dim=1)
            finalDenoisedReconstructionStateLoss = (initialSignalData - denoisedReconstructedData).pow(2).mean(dim=2).mean(dim=1)
            print("State Losses (VEF-D):", varReconstructionStateLoss.detach().mean().item(), encodingReconstructionStateLoss.detach().mean().item(), finalReconstructionStateLoss.detach().mean().item(), finalDenoisedReconstructionStateLoss.detach().mean().item())
            # Calculate the loss from taking other routes
            positionReconstructionLoss = (signalData - removedStampEncoding).pow(2).mean(dim=2).mean(dim=1)
            potentialVarReconstructionStateLoss = (signalData - potentialSignalData).pow(2).mean(dim=2).mean(dim=1)
            print("Path Losses (P-V2-S):", positionReconstructionLoss.detach().mean().item(), potentialVarReconstructionStateLoss.detach().mean().item(), signalEncodingLayerLoss.detach().mean().item())

            # Add up all the losses together.
            if 0.001 < potentialVarReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + potentialVarReconstructionStateLoss
            if 0.001 < finalReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + 0.25*finalReconstructionStateLoss
            if 0.001 < encodingReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + encodingReconstructionStateLoss
            if 0.001 < varReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + varReconstructionStateLoss
            if 0.001 < positionReconstructionLoss.mean():
                signalEncodingLoss = signalEncodingLoss + positionReconstructionLoss
            if 0.001 < signalEncodingLayerLoss.mean():
                signalEncodingLoss = signalEncodingLoss + signalEncodingLayerLoss

            if trainingFlag:
                # Accumulate the loss.
                self.accumulatedLoss = self.accumulatedLoss + finalDenoisedReconstructionStateLoss.mean()
                self.numAccumulations = self.numAccumulations + 1

                if self.accelerator.gradient_accumulation_steps <= self.numAccumulations:
                    self.trainingMethods.adjustNumEncodings(totalNumEncodings, self.accumulatedLoss / self.numAccumulations, forwardDirection)

                    # Reset the accumulation counter.
                    self.numAccumulations = 0
                    self.accumulatedLoss = 0

        # ------------------------------------------------------------------ #

        return encodedData, denoisedReconstructedData, signalEncodingLoss

        # ------------------------------------------------------------------ #  

    def reverseEncoding(self, decodedData, numSignalPath, signalEncodingLayerLoss, calculateLoss, trainingFlag):

        reversePath = []
        # Follow the path back to the original signal.
        for pathInd in range(len(numSignalPath) - 1, -1, -1):
            # Reconstruct to the current signal number in the path.
            decodedData, miniPath, signalEncodingLayerLoss \
                = self.encodeSignals(signalEncodingLayerLoss=signalEncodingLayerLoss,
                                     targetNumSignals=numSignalPath[pathInd],
                                     calculateLoss=calculateLoss,
                                     trainingFlag=trainingFlag,
                                     signalData=decodedData)
            reversePath.extend(miniPath)

        return decodedData, reversePath, signalEncodingLayerLoss

    def reconstructEncodedData(self, encodedData, numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False, trainingFlag=False):
        # If we are training, add noise to the final state to ensure continuity of the latent space.
        noisyEncodedData = self.encodeSignals.dataInterface.addNoise(encodedData, trainingFlag, noiseSTD=0.01)

        # Undo what was done in the initial adjustment.
        initialDecodedData = self.encodeSignals.finalVarianceInterface.unAdjustSignalVariance(noisyEncodedData)

        # Undo the signal encoding.
        decodedData, reversePath, signalEncodingLayerLoss = self.reverseEncoding(
            signalEncodingLayerLoss=signalEncodingLayerLoss,
            numSignalPath=numSignalForwardPath,
            decodedData=initialDecodedData,
            calculateLoss=calculateLoss,
            trainingFlag=trainingFlag,
        )
        # reconstructedInitEncodingData dimension: batchSize, numSignals, sequenceLength
        print("Signal Encoding Upward Path:", encodedData.size(1), reversePath, decodedData.size(1))
        assert reversePath[1:] == numSignalForwardPath[1:][::-1], f"Signal encoding path mismatch: {reversePath[1:]} != {numSignalForwardPath[1:][::-1]} reversed"

        # Learn how to remove positional encoding to each signal's position.
        noisyDecodedData = self.encodeSignals.dataInterface.addNoise(decodedData, trainingFlag, noiseSTD=0.001)
        reconstructedData = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(noisyDecodedData)

        # Denoise the final signals.
        denoisedReconstructedData = self.encodeSignals.denoiseSignals.applyDenoiser(reconstructedData)

        return initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss

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
