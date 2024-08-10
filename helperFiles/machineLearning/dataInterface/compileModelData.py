from sklearn.model_selection import train_test_split
import numpy as np
import torch
import copy
import os

# Import files for training and testing the model
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionPipeline import emotionPipeline
from ..modelControl.modelSpecifications.compileModelInfo import compileModelInfo  # Functions with model information
from .compileModelDataHelpers import compileModelDataHelpers

# Import interfaces for the model's data
from ...dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis
from ..featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames  # Functions to extract feature names
from ..modelControl.Models.pyTorch.Helpers.dataLoaderPyTorch import pytorchDataInterface

# Import interfaces for the metadata
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.emognitionInterface import emognitionInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.amigosInterface import amigosInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.dapperInterface import dapperInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.wesadInterface import wesadInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.caseInterface import caseInterface


class compileModelData(compileModelDataHelpers):

    def __init__(self, submodel, userInputParams, useTherapyData, accelerator=None):
        super().__init__(submodel, userInputParams, accelerator)
        # Initialize relevant classes.
        self.compileModelInfo = compileModelInfo()

        # Get the data folder.
        self.trainingFolder = os.path.dirname(__file__) + "/../../../" + self.compileModelInfo.getTrainingDataFolder(useTherapyData=useTherapyData)
        self.fullAnalysisSuffix = '_fullAnalysisParams'

        # Initialize the metadata interfaces.
        self.metaProtocolMap = {
            "emognition": emognitionInterface(),
            "amigos": amigosInterface(),
            "dapper": dapperInterface(),
            "wesad": wesadInterface(),
            "case": caseInterface()
        }

    # ------------------------ Compile Analysis Data ------------------------ #

    def compileProjectAnalysis(self, loadCompiledData=False, compiledModelName="compiledProjectTrainingInfo"):
        print(f"Reading in data for empatch")

        # Base case: we are loading in data that was already compiled.
        if loadCompiledData and os.path.isfile(f'{self.compiledInfoLocation}{compiledModelName}{self.compiledExtension}'): return self.loadCompiledInfo(compiledModelName)

        # Specify biomarker information.
        streamingOrder = self.compileModelInfo.streamingOrder  # A List Representing the Order of the Sensors being Streamed in.
        extractFeaturesFrom = self.compileModelInfo.streamingOrder  # A list with all the biomarkers from streamingOrder for feature extraction
        featureAverageWindows = self.compileModelInfo.featureAverageWindows  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
        plotTrainingData = False

        # Survey information.
        numQuestionOptions = [5] * 10  # PANAS Survey
        numQuestionOptions.extend([4] * 20)  # STAI Survey

        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        globalMetaAnalysisClass = globalMetaAnalysis(self.trainingFolder, surveyQuestions=[])
        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = globalMetaAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=False)

        # Compile experimental information.
        userNames = np.unique([i.split(" ")[-2].lower() for i in subjectOrder])
        subjectOrder = np.array([np.where(userName.split(" ")[-2].lower() == userNames)[0][0] for userName in subjectOrder])
        activityNames, activityLabels = self.compileModelInfo.extractActivityInformation(experimentalOrder, distinguishBaselines=False)

        # Compile the project data together
        data_to_store = {f"{compiledModelName}": [allRawFeatureIntervalTimes, allCompiledFeatureIntervals, surveyAnswersList, surveyQuestions, activityLabels, activityNames, numQuestionOptions, subjectOrder, featureNames]}
        data_to_store_Full = {
            f"{compiledModelName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals,
                                     allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes,
                                     subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalLabels, featureLabelTypes,
                                     featureNames, surveyQuestions, surveyAnswersList, surveyAnswerTimes, numQuestionOptions]}
        # Update the compiled data so the next person can use it.
        self.saveCompiledInfo(data_to_store_Full, compiledModelName + self.fullAnalysisSuffix)
        self.saveCompiledInfo(data_to_store, compiledModelName)

        return allRawFeatureIntervalTimes, allCompiledFeatureIntervals, surveyAnswersList, surveyQuestions, activityLabels, activityNames, numQuestionOptions, subjectOrder, featureNames

    def compileMetaAnalyses(self, metaDatasetNames, loadCompiledData=False, compiledModelName="compiledMetaTrainingInfo"):
        # Prepare to compile all the metadata analyses.
        metaSurveyQuestions, metaSurveyAnswersList, metaNumQuestionOptions, metaSubjectOrder = [], [], [], []
        metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaFeatureNames, metaActivityNames, metaActivityLabels = [], [], [], [], []

        plotTrainingData = False
        # For each meta-analysis protocol
        for metaDatasetName in metaDatasetNames:
            metaAnalysisProtocol = self.metaProtocolMap[metaDatasetName]

            # Prepare the data to go through the training interface.
            streamingOrder, biomarkerFeatureOrder, featureAverageWindows, featureNames, biomarkerFeatureNames = metaAnalysisProtocol.compileTrainingInfo()
            compiledModelFinalName = compiledModelName + f"_{metaAnalysisProtocol.datasetName}"
            numQuestionOptions = metaAnalysisProtocol.numQuestionOptions
            datasetName = metaAnalysisProtocol.datasetName

            print(f"Reading in metadata for {datasetName}")
            # Base case: we are loading in data that was already compiled.
            if loadCompiledData and os.path.isfile(f'{self.compiledInfoLocation}{compiledModelFinalName}{self.compiledExtension}'):
                allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswersList, activityNames, activityLabels = self.loadCompiledInfo(compiledModelFinalName)
            else:
                # Collected the training data.
                allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, \
                    allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
                    subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = metaAnalysisProtocol.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)

                # Compile experimental information.
                activityNames, activityLabels = metaAnalysisProtocol.extractExperimentLabels(experimentalOrder)

                # Update the compiled data so the next person can use it.
                data_to_store = {f"{compiledModelFinalName}": [allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswersList, activityNames, activityLabels]}
                data_to_store_Full = {f"{compiledModelFinalName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals,
                                                                    allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder,
                                                                    experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes,
                                                                    activityNames, activityLabels]}
                self.saveCompiledInfo(data_to_store_Full, compiledModelFinalName + self.fullAnalysisSuffix)
                self.saveCompiledInfo(data_to_store, compiledModelFinalName)

            # Organize all the metadata analyses.
            metaCompiledFeatureIntervals.append(allCompiledFeatureIntervals)
            metaRawFeatureTimeIntervals.append(allRawFeatureIntervalTimes)
            metaNumQuestionOptions.append(numQuestionOptions)
            metaSurveyAnswersList.append(surveyAnswersList)
            metaSurveyQuestions.append(surveyQuestions)
            metaActivityLabels.append(activityLabels)
            metaActivityNames.append(activityNames)
            metaFeatureNames.append(featureNames)
            metaSubjectOrder.append(subjectOrder)

        return metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames

    # -------------------- Machine Learning Preparation -------------------- #

    def compileModelsFull(self, metaDatasetNames, modelName, submodel, testSplitRatio, datasetNames, useFinalParams=False):
        # Compile the metadata together.
        metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, \
            metaSubjectOrder, metaFeatureNames, metaDatasetNames = self.compileMetaAnalyses(metaDatasetNames, loadCompiledData=True)

        # Compile the project data together
        allRawFeatureIntervalTimes, allCompiledFeatureIntervals, surveyAnswersList, surveyQuestions, activityLabels, activityNames, numQuestionOptions, subjectOrder, featureNames = self.compileProjectAnalysis(loadCompiledData=True)

        # Compile the meta-learning modules.
        allMetaModels, allMetaDataLoaders, allMetaLossDataHolders = self.compileModels(metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                                                                                       metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, metaTraining=True, specificInfo=None,
                                                                                       useFinalParams=useFinalParams, random_state=42)
        # Compile the final modules.
        allModels, allDataLoaders, allLossDataHolders = self.compileModels([allRawFeatureIntervalTimes], [allCompiledFeatureIntervals], [surveyAnswersList], [surveyQuestions], [activityLabels], [activityNames], [numQuestionOptions], [subjectOrder],
                                                                           [featureNames], datasetNames, modelName, submodel, testSplitRatio, metaTraining=False, specificInfo=None, useFinalParams=useFinalParams, random_state=42)
        # Create the meta-loss models and data loaders.
        allMetaLossDataHolders.extend(allLossDataHolders)

        return allModels, allDataLoaders, allLossDataHolders, allMetaModels, allMetaDataLoaders, allMetaLossDataHolders, metaDatasetNames

    def compileModels(self, metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                      metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, metaTraining, specificInfo=None, useFinalParams=False, random_state=42):
        # Initialize relevant holders.
        allModelPipelines = []
        lossDataHolders = []
        allDataLoaders = []

        # Specify model parameters
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = self.modelParameters.getModelInfo(submodel, specificInfo)
        print(f"\nSplitting the data into {'meta-' if metaTraining else ''}models:", flush=True)

        # For each meta-information collected.
        for metadataInd in range(len(metaRawFeatureTimeIntervals)):
            allRawFeatureTimeIntervals = metaRawFeatureTimeIntervals[metadataInd].copy()
            allCompiledFeatureIntervals = metaCompiledFeatureIntervals[metadataInd].copy()
            numQuestionOptions = metaNumQuestionOptions[metadataInd].copy()
            subjectOrder = np.asarray(metaSubjectOrder[metadataInd]).copy()
            surveyAnswersList = metaSurveyAnswersList[metadataInd].copy() - 1
            surveyQuestions = metaSurveyQuestions[metadataInd].copy()
            activityLabels = metaActivityLabels[metadataInd].copy()
            activityNames = metaActivityNames[metadataInd].copy()
            featureNames = metaFeatureNames[metadataInd].copy()
            metaDatasetName = metaDatasetNames[metadataInd]

            # Assert the assumptions made about the incoming data
            assert surveyAnswersList.min().item() >= -2, "All ratings must be greater than 0 (exception for -2, which is reserved for missing)."
            assert -1 not in surveyAnswersList, print("surveyAnswersList should contain ratings from 0 to n", flush=True)
            # Specify the incoming dimensions.
            # allCompiledFeatureIntervals dimension: batchSize, numSignals, sequenceLength*  -> *sequenceLength is not constant
            # allRawFeatureTimeIntervals dimension: batchSize, numSignals, sequenceLength*  -> *sequenceLength is not constant

            # ---------------------- Data Preparation ---------------------- #

            # Add the human activity recognition to the end.
            activityLabels = np.asarray(activityLabels, dtype=int).reshape(-1, 1)
            surveyAnswersList = np.hstack((surveyAnswersList, activityLabels))
            activityLabelInd = -1

            # Remove any experiments and signals that are bad.
            allSignalData, allSignalStopInds = self._padSignalData(allRawFeatureTimeIntervals, allCompiledFeatureIntervals)
            allSignalData, allSignalStopInds, allFeatureLabels, allSubjectInds = self._removeBadExperiments(allSignalData, allSignalStopInds, surveyAnswersList, subjectOrder)
            allSignalData, allSignalStopInds, featureNames = self._preprocessSignals(allSignalData, allSignalStopInds, featureNames)
            # allSignalData dimension: batchSize, numSignals, maxSequenceLength, [time, signal]
            # allFeatureLabels dimension: batchSize, numLabels
            # allSignalStopInds dimension: batchSize, numSignals
            # allSubjectInds dimension: batchSize
            # featureNames dimension: numSignals

            # Compile dataset-specific information.
            numExperiments, numSignals, maxSequenceLength, _ = allSignalData.size()
            allExperimentalIndices = torch.arange(0, numExperiments)
            numSubjects = max(allSubjectInds) + 1
            numLabels = allFeatureLabels.size(1)
            if numExperiments == 0: continue
            if numSignals == 0: continue

            # Organize the feature labels and identify any missing labels.
            allFeatureLabels, allSmallClassIndices = self.organizeLabels(allFeatureLabels, metaTraining, metaDatasetName, numSignals)
            # allSmallClassIndices dimension: numLabels, numSingleClassIndices
            # allFeatureLabels dimension: batchSize, numLabels

            # ---------------------- Test/Train Split ---------------------- #

            # Initiate a mask for distinguishing between training and testing data.
            currentTrainingMask = torch.full(allFeatureLabels.size(), fill_value=False, dtype=torch.bool)
            currentTestingMask = torch.full(allFeatureLabels.size(), fill_value=False, dtype=torch.bool)

            # For each type of label/emotion recorded.
            for labelTypeInd in range(numLabels):
                currentFeatureLabels = copy.deepcopy(allFeatureLabels[:, labelTypeInd])

                # Temporarily remove the single classes.
                singleClassIndices = allSmallClassIndices[labelTypeInd]
                currentFeatureLabels[singleClassIndices] = self.missingLabelValue
                # Remove the missing data from the arrays.
                missingClassMask = torch.isnan(currentFeatureLabels)
                currentIndices = np.delete(allExperimentalIndices.copy(), torch.nonzero(missingClassMask))

                if len(currentIndices) != 0:
                    # Get the number of unique classes for stratification.
                    stratifyBy = currentFeatureLabels[~missingClassMask]
                    minTestSplitRatio = len(np.unique(stratifyBy, return_counts=False)) / len(stratifyBy)

                    # Adjust the test split ratio if necessary.
                    if testSplitRatio < minTestSplitRatio:
                        print(f"\t\tWarning: The test split ratio is too small for the number of classes. Ignoring {surveyQuestions[labelTypeInd]} labels", flush=True)
                        continue

                    # Randomly split the data and labels, keeping a balance between testing/training.
                    Training_Indices, Testing_Indices = train_test_split(currentIndices, test_size=max(minTestSplitRatio, testSplitRatio),
                                                                         shuffle=True, stratify=stratifyBy, random_state=random_state)

                    # Populate the training and testing mask                        
                    currentTestingMask[:, labelTypeInd][Testing_Indices] = True
                    currentTrainingMask[:, labelTypeInd][Training_Indices] = True
                    # Add in the single class values
                    currentTrainingMask[:, labelTypeInd][singleClassIndices] = True

            # ---------------------- Data Adjustments ---------------------- #

            # Remove any unused activity labels/names.
            goodActivityMask = (currentTestingMask[:, activityLabelInd]) | (currentTrainingMask[:, activityLabelInd])
            activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask] \
                = self.organizeActivityLabels(activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask])  # Expects inputs/outputs from 0 to n-1
            allFeatureLabels[~goodActivityMask] = -1  # Remove any unused activity indices (as the good indices were rehashed)

            # Data augmentation to increase the variations of datapoints.
            if metaDatasetName.lower() not in self.dontShiftDatasets:
                augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds \
                    = self.addShiftedSignals(allSignalData, allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds)
            else:
                allFeatureData = self.dataInterface.getRecentSignalPoints(allSignalData, self.maxSeqLength + self.numSecondsShift)

                augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds \
                    = self.dataInterface.getRecentSignalPoints(allFeatureData, self.maxSeqLength), allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds

            # Add the demographic information.
            augmentedFeatureData, numSubjectIdentifiers, demographicLength = self.addDemographicInfo(augmentedFeatureData, augmentedSubjectInds, metadataInd)

            # ---------------------- Create the Model ---------------------- #

            # Get the model parameters
            batch_size = self.modelParameters.getTrainingBatchSize(submodel, numExperiments=len(augmentedFeatureData))

            # Organize the training data into the expected pytorch format.
            pytorchDataClass = pytorchDataInterface(batch_size=batch_size, num_workers=0, shuffle=True, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask)

            # Initialize and train the model class.
            modelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=metaDatasetName, modelName=modelName, allEmotionClasses=numQuestionOptions.copy(),
                                            sequenceLength=sequenceLength, maxNumSignals=numSignals, numSubjectIdentifiers=numSubjectIdentifiers, demographicLength=demographicLength, numSubjects=numSubjects,
                                            userInputParams=self.userInputParams, emotionNames=surveyQuestions, activityNames=activityNames, featureNames=featureNames, submodel=submodel, useFinalParams=useFinalParams, debuggingResults=True)

            # Hugging face integration.
            modelDataLoader = modelPipeline.acceleratorInterface(modelDataLoader)

            trainingInformation = modelPipeline.getDistributedModels(model=None, submodel="trainingInformation")
            trainingInformation.addSubmodel(submodel)

            # Store the information.
            allDataLoaders.append(modelDataLoader)
            allModelPipelines.append(modelPipeline)

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=True)

        # Prepare loss data holders.
        for modelInd in range(len(allModelPipelines)):
            dataLoader = allDataLoaders[modelInd]

            # Organize the training data into the expected pytorch format.
            numExperiments, numSignals, signalDimension = dataLoader.dataset.getSignalInfo()
            pytorchDataClass = pytorchDataInterface(batch_size=self.modelParameters.getInferenceBatchSize(submodel, numSignals), num_workers=0, shuffle=False, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(*dataLoader.dataset.getAll())

            # Store the information.
            lossDataHolders.append(modelDataLoader)

        return allModelPipelines, allDataLoaders, lossDataHolders

    def onlyPreloadModelAttributes(self, modelName, datasetNames, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, allDummyModelPipelines=()):
        # Initialize relevant holders.
        userInputParams = {'deviceListed': "cpu", 'submodel': 'emotionPrediction', 'numExpandedSignals': 2, 'numSigEncodingLayers': 4, 'numSigLiftedChannels': 4,
                           'compressionFactor': 1.5, 'expansionFactor': 1.5, 'numInterpreterHeads': 4, 'numBasicEmotions': 8, 'sequenceLength': 240}

        # Overwrite the location of the saved models.
        self.modelMigration.replaceFinalModelFolder("_finalModels/storedModels/")

        if len(allDummyModelPipelines) == 0:
            print(f"\nInitializing the models for {modelName}:", flush=True)
            # For each model we want.
            for metadataInd in range(len(datasetNames)):
                datasetName = datasetNames[metadataInd]

                # Initialize and train the model class.
                dummyModelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=datasetName, modelName=modelName, allEmotionClasses=[],
                                                     sequenceLength=240, maxNumSignals=500, numSubjectIdentifiers=1, demographicLength=0, numSubjects=1, userInputParams=userInputParams,
                                                     emotionNames=[], activityNames=[], featureNames=[], submodel=loadSubmodel, useFinalParams=True, debuggingResults=True)
                # Hugging face integration.
                dummyModelPipeline.acceleratorInterface()

                # Store the information.
                allDummyModelPipelines.append(dummyModelPipeline)

        # Load in the model attributes.
        self.modelMigration.loadModels(allDummyModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=False)

        return allDummyModelPipelines
