import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class sharedSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, operatorType, encodedDimension, numSharedEncoderLayers, learningProtocol, neuralOperatorParameters):
        super(sharedSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numInputSignals=1, numOutputSignals=1, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = neuralOperatorParameters  # The parameters for the neural operator.
        self.encodedTimeWindow = modelConstants.modelTimeWindow  # The time window for the encoded signal.
        self.numSharedEncoderLayers = numSharedEncoderLayers  # The number of shared encoder layers.
        self.learningProtocol = learningProtocol  # The learning protocol for the model.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.

        # Initialize the pseudo-encoded times for the fourier data.
        hyperSampledTimes = torch.linspace(start=0, end=self.encodedTimeWindow, steps=self.encodedDimension).flip(dims=[0])
        self.register_buffer(name='hyperSampledTimes', tensor=hyperSampledTimes)  # Non-learnable parameter.
        deltaTimes = torch.unique(self.hyperSampledTimes.diff().round(decimals=4))
        assert len(deltaTimes) == 1, f"The time gaps are not similar: {deltaTimes}"

        # The neural layers for the signal encoder.
        self.healthGenerationModel = self.healthGeneration(numOutputFeatures=encodedDimension)
        self.processingLayers, self.neuralLayers = nn.ModuleList(), nn.ModuleList()
        self.healthProfileJacobians = self.initializeJacobianParams(1)
        for _ in range(self.numSharedEncoderLayers): self.addLayer()

    def forward(self):
        raise "You cannot call the dataset-specific signal encoder module."

    def addLayer(self):
        self.neuralLayers.append(self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True))
        if self.learningProtocol == 'rCNN': self.processingLayers.append(self.postProcessingLayerRCNN(numSignals=1, sequenceLength=self.encodedDimension))
        elif self.learningProtocol == 'FC': self.processingLayers.append(self.postProcessingLayerFC(sequenceLength=self.encodedDimension))
        elif self.learningProtocol == 'CNN': self.processingLayers.append(self.postProcessingLayerCNN(numSignals=self.numSignals))
        else: raise "The learning protocol is not yet implemented."

    # Learned up-sampling of the health profile.
    def generateHealthProfile(self, healthProfile):
        healthProfile = self.healthGenerationModel(healthProfile.unsqueeze(1)).squeeze(1)
        healthProfile = self.applyManifoldScale(healthProfile, self.healthProfileJacobians) # TODO: Add back

        return healthProfile

    def learningInterface(self, layerInd, signalData):
        # Extract the signal data parameters.
        batchSize, numSignals, signalLength = signalData.shape
        signalData = signalData.view(batchSize*numSignals, 1, signalLength)

        # For the forward/harder direction.
        if not reversibleInterface.forwardDirection:
            # Apply the neural operator layer with activation.
            signalData = self.neuralLayers[layerInd](signalData)
            signalData = self.processingLayers[layerInd](signalData)
        else:
            # Get the reverse layer index.
            pseudoLayerInd = len(self.neuralLayers) - layerInd - 1
            assert 0 <= pseudoLayerInd < len(self.neuralLayers), f"The pseudo layer index is out of bounds: {pseudoLayerInd}, {len(self.neuralLayers)}, {layerInd}"

            # Apply the neural operator layer with activation.
            signalData = self.processingLayers[pseudoLayerInd](signalData)
            signalData = self.neuralLayers[pseudoLayerInd](signalData)

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
        numJacobians = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and 'healthProfileJacobian' in name)
        numParams = sum(p.numel() for name, p in self.named_parameters()) - numJacobians - numInitParams

        # Print the number of trainable parameters.
        totalParams = sum(p.numel() for name, p in self.named_parameters())
        print(f'The model has {totalParams} trainable parameters: {numJacobians} jacobians, {numParams} meta-weights, and {numInitParams} initial weights.')


if __name__ == "__main__":
    # General parameters.
    _neuralOperatorParameters = modelParameters.getNeuralParameters({'waveletType': 'bior3.1'})['neuralOperatorParameters']
    _batchSize, _numSignals, _sequenceLength = 1, 1, 256
    modelConstants.profileDimension = 64
    _numSharedEncoderLayers = 1

    # Set up the parameters.
    neuralLayerClass = sharedSignalEncoderModel(operatorType='wavelet', encodedDimension=_sequenceLength, numSharedEncoderLayers=_numSharedEncoderLayers, learningProtocol='rCNN', neuralOperatorParameters=_neuralOperatorParameters)
    neuralLayerClass.printParams()
