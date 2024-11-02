# Import files for machine learning
import torch

from .emotionModel.emotionModelHead import emotionModelHead
from .emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionModel.emotionModelHelpers.generalMethods.dataAugmentation import dataAugmentation
from .emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from .emotionModel.emotionModelHelpers.lossInformation.organizeTrainingLosses import organizeTrainingLosses
from .emotionModel.emotionModelHelpers.modelConstants import modelConstants
from .emotionModel.emotionModelHelpers.modelVisualizations.modelVisualizations import modelVisualizations
from .emotionModel.emotionModelHelpers.optimizerMethods.optimizerMethods import optimizerMethods


class emotionPipelineHelpers:

    def __init__(self, accelerator, datasetName, allEmotionClasses, numSubjects, userInputParams,
                 emotionNames, activityNames, featureNames, submodel, numExperiments):
        # General parameters.
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
        self.model = emotionModelHead(submodel=submodel, userInputParams=userInputParams, emotionNames=emotionNames, activityNames=activityNames,
                                      featureNames=featureNames, numSubjects=numSubjects, datasetName=datasetName, numExperiments=numExperiments)
        # try: self.model = torch.compile(self.model, backend='inductor')  # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        # # self.model = torch.jit.script(self.model)
        # except Exception as e: print(f"\t\tCannot use torch compilation yet: {e}")

        # Initialize helper classes.
        self.organizeLossInfo = organizeTrainingLosses(accelerator, allEmotionClasses, self.activityLabelInd)
        self.modelVisualization = modelVisualizations(accelerator, modelSubfolder="trainingFigures/")
        self.optimizerMethods = optimizerMethods(userInputParams)
        self.dataInterface = emotionDataInterface()
        self.dataAugmentation = dataAugmentation()
        self.modelHelpers = modelHelpers()

        # Finish setting up the model.
        if submodel == modelConstants.emotionModel:
            self.model.sharedEmotionModel.lastActivityLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.activityClass_lossType, predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.
            self.model.sharedEmotionModel.lastEmotionLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.emotionDist_lossType, predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.

        # Assert data integrity of the inputs.
        assert len(self.emotionNames) == len(self.allEmotionClasses), f"Found {len(self.emotionNames)} emotions with {len(self.allEmotionClasses)} classes specified."
        assert len(self.activityNames) == self.numActivities, f"Found {len(self.activityNames)} activities with {self.numActivities} classes specified."

    def compileOptimizer(self, submodel):
        # Initialize the optimizer and scheduler.
        self.optimizer, self.scheduler = self.optimizerMethods.addOptimizer(submodel, self.model)

    def acceleratorInterface(self, dataLoader=None):
        if dataLoader is None: self.optimizer, self.scheduler, self.model = self.accelerator.prepare(self.optimizer, self.scheduler, self.model)
        else: self.optimizer, self.scheduler, self.model, dataLoader = self.accelerator.prepare(self.optimizer, self.scheduler, self.model, dataLoader)

        return dataLoader

    @staticmethod
    def prepareInformation(dataLoader):
        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
        allSignalData, allSignalIdentifiers, allMetadata = emotionDataInterface.separateData(allData)

        # Assert the integrity of the dataloader.
        assert allLabels.shape == allTrainingMasks.shape, "We should specify the training indices for each label"
        assert allLabels.shape == allTestingMasks.shape, "We should specify the testing indices for each label"

        return allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata

    # ------------------------------------------------------------------ #

    def getTrainingEpoch(self, submodel):
        if submodel == modelConstants.signalEncoderModel: return max(0, len(self.model.specificSignalEncoderModel.trainingLosses_signalReconstruction) - 1)
        elif submodel == modelConstants.emotionModel: return max(0, len(self.model.specificEmotionModel.trainingLosses_signalReconstruction) - 1)
        else: raise Exception()

    def setupTraining(self, submodel, trainSharedLayers=False, inferenceTraining=False, profileTraining=False):
        # Do not train the model at all.
        self.setupTrainingFlags(self.model, trainingFlag=False)

        # Profile training.
        if profileTraining:
            self.model.specificSignalEncoderModel.physiologicalProfileAnsatz.requires_grad = True
            assert not trainSharedLayers, "We cannot train layers during profile training."
            assert not inferenceTraining, "We cannot train layers during profile training."
            return None

        if inferenceTraining:
            # Label the model we are training.
            self.setupTrainingFlags(self.model.inferenceModel, trainingFlag=True)
            assert not trainSharedLayers, "We cannot train layers during inference."
            assert not profileTraining, "We cannot train layers during inference."

        # Emotion model training.
        if submodel == modelConstants.emotionModel:
            if trainSharedLayers: self.setupTrainingFlags(self.model.sharedEmotionModel, trainingFlag=True)
            self.setupTrainingFlags(self.model.specificEmotionModel, trainingFlag=True)
        else:
            # Signal encoder training
            if trainSharedLayers: self.setupTrainingFlags(self.model.sharedSignalEncoderModel, trainingFlag=True)
            self.setupTrainingFlags(self.model.specificSignalEncoderModel, trainingFlag=True)

    @staticmethod
    def setupTrainingFlags(model, trainingFlag):
        # Set the model's training mode.
        if trainingFlag: model.train()
        else: model.eval()

        # Update requires_grad for all parameters
        for param in model.parameters():
            param.requires_grad = trainingFlag
