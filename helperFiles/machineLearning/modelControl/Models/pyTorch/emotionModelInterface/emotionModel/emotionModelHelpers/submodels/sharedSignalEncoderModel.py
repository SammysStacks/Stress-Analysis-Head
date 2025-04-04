import copy

import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, numLayers, learningProtocol, neuralOperatorParameters):
        super(sharedSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=1, numOutputSignals=1, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.encodedTimeWindow = modelConstants.modelTimeWindow  # The time window for the encoded signal.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numLayers = numLayers  # The number of shared encoder layers.

        numIgnoredSharedHF = modelConstants.userInputParams['numIgnoredSharedHF']
        # Only apply a transformation to the lowest of the high frequency decompositions.
        self.neuralOperatorParameters['wavelet']['encodeHighFrequencyProtocol'] = f'highFreq-{0}-{numIgnoredSharedHF}'

        # Initialize the pseudo-encoded times for the fourier data.
        hyperSampledTimes = torch.linspace(start=0, end=self.encodedTimeWindow, steps=self.encodedDimension).flip(dims=[0])
        self.register_buffer(name='hyperSampledTimes', tensor=hyperSampledTimes)  # Non-learnable parameter.
        deltaTimes = torch.unique(self.hyperSampledTimes.diff().round(decimals=4))
        assert len(deltaTimes) == 1, f"The time gaps are not similar: {deltaTimes}"

        # The neural layers for the signal encoder.
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True)
        self.healthGenerationModel = self.healthGeneration()
        self.fourierModel = self.fourierAdjustments()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    # Learned up-sampling of the health profile.
    def generateHealthProfile(self, healthProfile):
        healthProfile = self.healthGenerationModel(healthProfile.unsqueeze(1)).squeeze(1)
        return healthProfile

    def learningInterface(self, signalData, compilingFunction):
        # Extract the signal data parameters.
        batchSize, numSignals, signalLength = signalData.shape
        signalData = signalData.view(batchSize*numSignals, 1, signalLength)

        # Apply the neural operator layer with activation.
        self.neuralLayers.compilingFunction = compilingFunction
        signalData = self.neuralLayers(signalData)
        self.neuralLayers.compilingFunction = None

        # Reshape the signal data.
        signalData = signalData.view(batchSize, numSignals, signalLength)

        return signalData.contiguous()

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

    def interpolateOriginalSignals(self, signalData, resampledSignalData):
        # Extract the dimensions of the data.
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        batchSize, numSignals, encodedDimension = resampledSignalData.size()

        # Align the timepoints to the physiological times.
        reversedPhysiologicalTimes = torch.flip(self.hyperSampledTimes, dims=[0])
        mappedPhysiologicalTimedInds = encodedDimension - 1 - torch.searchsorted(sorted_sequence=reversedPhysiologicalTimes, input=timepoints, out=None, out_int32=False, right=False)  # timepoints <= relativeTimesExpanded[mappedPhysiologicalTimedInds]
        # Ensure the indices don't exceed the size of the last dimension of reconstructedSignalData.
        validIndsRight = torch.clamp(mappedPhysiologicalTimedInds, min=0, max=encodedDimension - 1)  # relativeTimesExpanded[validIndsLeft] < timepoints
        validIndsLeft = torch.clamp(mappedPhysiologicalTimedInds + 1, min=0, max=encodedDimension - 1)  # timepoints <= relativeTimesExpanded[validIndsRight]
        # mappedPhysiologicalTimedInds dimension: batchSize, numSignals, maxSequenceLength

        # Get the closest physiological data to the timepoints.
        relativeTimesExpanded = self.hyperSampledTimes.view(1, 1, -1).expand_as(resampledSignalData)
        closestPhysiologicalDataLeft = torch.gather(input=resampledSignalData, dim=2, index=validIndsLeft)
        closestPhysiologicalDataRight = torch.gather(input=resampledSignalData, dim=2, index=validIndsRight)
        closestPhysiologicalTimesLeft = torch.gather(input=relativeTimesExpanded, dim=2, index=validIndsLeft)
        closestPhysiologicalTimesRight = torch.gather(input=relativeTimesExpanded, dim=2, index=validIndsRight)
        assert ((closestPhysiologicalTimesLeft <= timepoints + 0.1) & (timepoints - 0.1 <= closestPhysiologicalTimesRight)).all(), "The timepoints must be within the range of the closest physiological times."
        # closestPhysiologicalData dimension: batchSize, numSignals, maxSequenceLength

        # Perform linear interpolation.
        linearSlopes = (closestPhysiologicalDataRight - closestPhysiologicalDataLeft) / (closestPhysiologicalTimesRight - closestPhysiologicalTimesLeft).clamp(min=1e-20)
        linearSlopes[closestPhysiologicalTimesLeft == closestPhysiologicalTimesRight] = 0

        # Calculate the error in signal reconstruction (encoding loss).
        interpolatedData = closestPhysiologicalDataLeft + (timepoints - closestPhysiologicalTimesLeft) * linearSlopes

        return interpolatedData

    def printParams(self):
        # Count the trainable parameters.
        numInitParams = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and 'healthGenerationModel' in name)
        numParams = sum(p.numel() for name, p in self.named_parameters()) - numInitParams

        # Print the number of trainable parameters.
        totalParams = sum(p.numel() for name, p in self.named_parameters())
        print(f'The model has {totalParams} trainable parameters: {numParams} meta-weights, and {numInitParams} initial weights.')


if __name__ == "__main__":
    # General parameters.
    _neuralOperatorParameters = modelParameters.getNeuralParameters({'waveletType': 'bior3.1'})['neuralOperatorParameters']
    modelConstants.userInputParams['profileDimension'] = 64
    _batchSize, _numSignals, _sequenceLength = 1, 1, 256
    _numSharedEncoderLayers = 4

    # Set up the parameters.
    neuralLayerClass = sharedSignalEncoderModel(operatorType='wavelet', encodedDimension=_sequenceLength, numLayers=_numSharedEncoderLayers, learningProtocol='reversibleLieLayer', neuralOperatorParameters=_neuralOperatorParameters)
    neuralLayerClass.printParams()
