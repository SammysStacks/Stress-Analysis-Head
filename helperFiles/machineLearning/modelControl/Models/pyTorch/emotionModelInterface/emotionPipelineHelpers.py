from .emotionModel.emotionModel import emotionModel
from .emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionModel.emotionModelHelpers.generalMethods.dataAugmentation import dataAugmentation
from .emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from .emotionModel.emotionModelHelpers.lossInformation.organizeTrainingLosses import organizeTrainingLosses
from .emotionModel.emotionModelHelpers.modelConstants import modelConstants
from .emotionModel.emotionModelHelpers.modelParameters import modelParameters
from .emotionModel.emotionModelHelpers.modelVisualizations.modelVisualizations import modelVisualizations
from .emotionModel.emotionModelHelpers.optimizerMethods.optimizerMethods import optimizerMethods


class emotionPipelineHelpers:

    def __init__(self, accelerator, datasetName, allEmotionClasses, numSubjects,
                 emotionNames, activityNames, featureNames, submodel, numExperiments):
        # General parameters.
        self.allEmotionClassWeights = None  # The class weights for the emotion classes.
        self.activityClassWeights = None  # The class weights for the activity classes.
        self.accelerator = accelerator  # Hugging face interface to speed up the training process.
        self.optimizer = None  # The optimizer for the model.
        self.scheduler = None  # The learning rate scheduler for the model.
        self.model = None  # The model being used for the training process.

        # Dataset-specific parameters.
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: [numEmotions]
        self.activityLabelInd = len(emotionNames)  # The index of the activity label in the label array.
        self.numActivities = len(activityNames)  # The number of activities we are predicting. Type: int
        self.numEmotions = len(emotionNames)  # The number of emotions we are predicting. Type: int
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.featureNames = featureNames  # The names of each signal in the model. Dim: numSignals
        self.datasetName = datasetName  # The name of the specific dataset being used in this model (case, wesad, etc.)

        # Initialize the emotion model.
        self.model = emotionModel(submodel=submodel, emotionNames=emotionNames, activityNames=activityNames, featureNames=featureNames, allEmotionClasses=allEmotionClasses, numSubjects=numSubjects, datasetName=datasetName, numExperiments=numExperiments)
        self.model = self.model.double()  # Convert the model to double precision.

        # if 'HPC' not in modelConstants.userInputParams['deviceListed']:
        #     # Speed up the model training process, if possible.
        #     try: self.model = torch.compile(self.model, backend='inductor')  # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        #     except Exception as e: print(f"\t\tCannot use torch compilation yet: {e}")

        # Initialize helper classes.
        self.organizeLossInfo = organizeTrainingLosses(accelerator, allEmotionClasses, numActivities=self.numActivities, activityLabelInd=self.activityLabelInd)
        self.modelVisualization = modelVisualizations(accelerator, datasetName=datasetName, activityLabelInd=self.activityLabelInd)
        self.optimizerMethods = optimizerMethods()
        self.dataAugmentation = dataAugmentation()
        self.modelHelpers = modelHelpers()

        # Assert data integrity of the inputs.
        assert len(self.emotionNames) == len(self.allEmotionClasses), f"Found {len(self.emotionNames)} emotions with {len(self.allEmotionClasses)} classes specified."
        assert len(self.activityNames) == self.numActivities, f"Found {len(self.activityNames)} activities with {self.numActivities} classes specified."

    def assignClassWeights(self, signalLabels, trainingLabelMask, testingLabelMask):
        self.allEmotionClassWeights, self.activityClassWeights = self.organizeLossInfo.getClassWeights(signalLabels, trainingLabelMask, testingLabelMask, self.numActivities)

    # ------------------------------------------------------------------ #

    def resetPhysiologicalProfile(self, submodel):
        numEpochs = self.getTrainingEpoch(submodel) + 1  # +1 to account for the current epoch.

        # Get the current number of epochs for the profile model.
        if submodel != modelConstants.signalEncoderModel: numProfileShots = modelParameters.getProfileEpochs()
        else: numProfileShots = min(max(3, numEpochs), modelParameters.getProfileEpochs())

        # Reset and get the parameters that belong to the profile model
        self.model.specificSignalEncoderModel.profileModel.resetProfileHolders(numProfileShots)
        if 4 < modelParameters.getProfileEpochs(): self.model.specificSignalEncoderModel.profileModel.resetProfileWeights()

        return numProfileShots

    def compileOptimizer(self, submodel):
        # Initialize the optimizer and scheduler.
        self.optimizer, self.scheduler = self.optimizerMethods.addOptimizer(submodel, self.model)

    def acceleratorInterface(self, dataLoader=None):
        if dataLoader is None: self.optimizer, self.scheduler, self.model = self.accelerator.prepare(self.optimizer, self.scheduler, self.model)
        else: self.optimizer, self.scheduler, self.model, dataLoader = self.accelerator.prepare(self.optimizer, self.scheduler, self.model, dataLoader)
        return dataLoader

    # ------------------------------------------------------------------ #

    def getTrainingEpoch(self, submodel):
        if submodel == modelConstants.signalEncoderModel: return max(0, len(self.model.specificSignalEncoderModel.trainingLosses_signalReconstruction) - 1)
        elif submodel == modelConstants.emotionModel: return max(0, len(self.model.specificEmotionModel.trainingLosses_signalReconstruction) - 1)
        else: raise Exception()

    def setupTraining(self, submodel, profileTraining, specificTraining, trainSharedLayers):
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Set the model to evaluation mode.

        # Emotion model training.
        if submodel == modelConstants.emotionModel:
            # Activity model training.
            self.setupTrainingFlags(self.model.specificActivityModel, trainingFlag=specificTraining)
            self.setupTrainingFlags(self.model.sharedActivityModel, trainingFlag=trainSharedLayers)

            # Emotion model training.
            self.setupTrainingFlags(self.model.specificEmotionModel, trainingFlag=specificTraining)
            self.setupTrainingFlags(self.model.sharedEmotionModel, trainingFlag=trainSharedLayers)
            assert not profileTraining, "We cannot train layers during emotion model training."
        else:
            # Signal encoder training.
            self.setupTrainingFlags(self.model.sharedSignalEncoderModel, trainingFlag=trainSharedLayers)
            self.setupTrainingFlags(self.model.specificSignalEncoderModel, trainingFlag=specificTraining)
            self.setupTrainingFlags(self.model.specificSignalEncoderModel.profileModel, trainingFlag=profileTraining)

    @staticmethod
    def setupTrainingFlags(model, trainingFlag):
        # Set the model's training mode.
        if trainingFlag: model.train()
        else: model.eval()

        # Update requires_grad for all parameters
        for param in model.parameters():
            param.requires_grad = trainingFlag

    # ------------------------------------------------------------------ #

    @staticmethod
    def prepareInformation(dataLoader):
        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()

        # Separate the data and mask information.
        allTrainingLabelMask, allTrainingSignalMask = emotionDataInterface.separateMaskInformation(allTrainingMasks, allLabels.size(-1))
        allTestingLabelMask, allTestingSignalMask = emotionDataInterface.separateMaskInformation(allTestingMasks, allLabels.size(-1))
        allSignalData, allSignalIdentifiers, allMetadata = emotionDataInterface.separateData(allData)
        # allSignalData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # allTrainingLabelMask, allTestingLabelMask: batchSize, numEmotions + 1 (activity)
        # allTrainingSignalMask, allTestingSignalMask: batchSize, numSignals
        # allSignalIdentifiers: batchSize, numSignals, numSignalIdentifiers
        # allLabels: batchSize, numEmotions + 1 (activity) + numSignals
        # allMetadata: batchSize, numMetadata

        return allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask

    def extractBatchInformation(self, batchData):
        # Extract the data, labels, and testing/training indices.
        batchSignalInfo, batchSignalLabels, batchTrainingMask, batchTestingMask = batchData
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.accelerator.device), batchTestingMask.to(self.accelerator.device)
        batchSignalInfo, batchSignalLabels = batchSignalInfo.to(self.accelerator.device), batchSignalLabels.to(self.accelerator.device)

        # Separate the mask information.
        batchTrainingLabelMask, batchTrainingSignalMask = emotionDataInterface.separateMaskInformation(batchTrainingMask, batchSignalLabels.size(-1))
        batchTestingLabelMask, batchTestingSignalMask = emotionDataInterface.separateMaskInformation(batchTestingMask, batchSignalLabels.size(-1))
        return batchSignalInfo, batchSignalLabels, batchTrainingLabelMask, batchTestingLabelMask, batchTrainingSignalMask, batchTestingSignalMask
