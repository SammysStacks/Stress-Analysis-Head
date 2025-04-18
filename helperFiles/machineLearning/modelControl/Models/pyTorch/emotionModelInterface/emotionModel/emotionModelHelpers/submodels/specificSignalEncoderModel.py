import copy

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.profileModel import profileModel


class specificSignalEncoderModel(neuralOperatorInterface):

    def __init__(self, numExperiments, operatorType, encodedDimension, featureNames, numLayers, neuralOperatorParameters):
        super(specificSignalEncoderModel, self).__init__(operatorType=operatorType, sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=len(featureNames), numOutputSignals=len(featureNames), addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.encodedDimension = encodedDimension  # The dimension of the encoded signal.
        self.numExperiments = numExperiments  # The number of experiments.
        self.numSignals = len(featureNames)  # The number of signals to encode.
        self.featureNames = featureNames  # The names of the signals to encode.
        self.numLayers = numLayers  # The number of specific encoder layers.

        # Set the wavelet parameters.
        self.neuralOperatorParameters['wavelet']['minWaveletDim'] = encodedDimension // 2

        # The neural layers for the signal encoder.
        self.profileModel = profileModel(numExperiments=numExperiments, numSignals=self.numSignals, encodedDimension=encodedDimension)
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters)

        # Assert the validity of the input parameters.
        assert self.encodedDimension % 2 == 0, "The encoded dimension must be divisible by 2."
        assert 0 < self.encodedDimension, "The encoded dimension must be greater than 0."

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction, self.testingLosses_signalReconstruction = None, None
        self.givensAnglesFeaturesPath, self.normalizationFactorsPath = None, None
        self.activationParamsPath, self.numFreeParams = None, None
        self.resetModel()

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def healthEmbeddingModel(self, batchInds, fourierModel):
        return self.profileModel.getHealthEmbedding(batchInds, fourierModel)

    def resetModel(self):
        # Signal encoder reconstruction holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: loadSubmodelEpochs, numTrainingSignals
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: loadSubmodelEpochs, numTestingSignals
        self.givensAnglesFeaturesPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*, numParams
        self.activationParamsPath = []  # List of activation bounds. Dim: loadSubmodelEpochs, numActivations, numActivationParams
        self.normalizationFactorsPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*
        self.numFreeParams = []  # List of the number of free parameters. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*

    def learningInterface(self, signalData, compilingFunction):
        # Apply the neural operator layer with activation.
        self.neuralLayers.compilingFunction = compilingFunction
        signalData = self.neuralLayers(signalData)
        self.neuralLayers.compilingFunction = None

        return signalData.contiguous()

    def printParams(self):
        # Count the trainable parameters.
        numProfileParams = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad and 'profileModel' in name) / self.numExperiments
        numParams = (sum(p.numel() for name, p in self.named_parameters()) - numProfileParams) / self.numSignals

        # Print the number of trainable parameters.
        totalParams = numParams*self.numSignals + numProfileParams*self.numExperiments
        print(f'The model has {totalParams} trainable parameters: {numProfileParams + numParams} split between {numParams} per signal and {numProfileParams} per experiment.')


if __name__ == "__main__":
    # General parameters.
    _neuralOperatorParameters = modelParameters.getNeuralParameters({'waveletType': 'bior3.1'})['neuralOperatorParameters']
    modelConstants.userInputParams['initialProfileAmp'] = 1e-3
    modelConstants.userInputParams['profileDimension'] = 64
    _batchSize, _numSignals, _sequenceLength = 1, 1, 256
    _numSpecificEncoderLayers = 1

    # Set up the parameters.
    _featureNames = [f"signal_{i}" for i in range(_numSignals)]
    neuralLayerClass = specificSignalEncoderModel(numExperiments=_batchSize, operatorType='wavelet', encodedDimension=_sequenceLength, featureNames=_featureNames, numLayers=_numSpecificEncoderLayers, neuralOperatorParameters=_neuralOperatorParameters)
    neuralLayerClass.printParams()
