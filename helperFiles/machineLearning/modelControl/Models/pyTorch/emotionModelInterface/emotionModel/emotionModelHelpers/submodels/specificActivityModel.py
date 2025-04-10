import copy

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.neuralOperatorInterface import neuralOperatorInterface


class specificActivityModel(neuralOperatorInterface):

    def __init__(self, encodedDimension, activityNames, numLayers, neuralOperatorParameters):
        super(specificActivityModel, self).__init__(operatorType='fourier', sequenceLength=encodedDimension, numLayers=numLayers, numInputSignals=1, numOutputSignals=1, addBiasTerm=False)
        # General model parameters.
        self.neuralOperatorParameters = copy.deepcopy(neuralOperatorParameters)  # The parameters for the neural operator.
        self.numActivities = len(activityNames)  # The number of activities.
        self.activityNames = activityNames  # The names of the activities.
        self.numLayers = numLayers  # The number of layers in the model.

        # Set the wavelet parameters.
        self.neuralOperatorParameters['wavelet']['minWaveletDim'] = encodedDimension // 2

        # The neural layers for the signal encoder.
        self.neuralLayers = self.getNeuralOperatorLayer(neuralOperatorParameters=self.neuralOperatorParameters, reversibleFlag=True)

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction, self.testingLosses_signalReconstruction = None, None
        self.givensAnglesFeaturesPath, self.normalizationFactorsPath = None, None
        self.activationParamsPath, self.numFreeParams = None, None
        self.resetModel()

    def forward(self): raise "You cannot call the dataset-specific signal encoder module."

    def resetModel(self):
        # Signal encoder reconstruction holders.
        self.trainingLosses_signalReconstruction = []  # List of list of data reconstruction training losses. Dim: loadSubmodelEpochs, numTrainingSignals
        self.testingLosses_signalReconstruction = []  # List of list of data reconstruction testing losses. Dim: loadSubmodelEpochs, numTestingSignals
        self.givensAnglesFeaturesPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*, numParams
        self.activationParamsPath = []  # List of activation bounds. Dim: loadSubmodelEpochs, numActivations, numActivationParams
        self.normalizationFactorsPath = []  # List of Givens angles. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*
        self.numFreeParams = []  # List of the number of free parameters. Dim: loadSubmodelEpochs, numModuleLayers, *numSignals*

    def learningInterface(self, signalData):
        # Apply the neural operator layer with activation.
        return self.neuralLayers(signalData).contiguous()
