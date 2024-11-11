from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import torch
import os

from ..modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ..modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionPipeline import emotionPipeline
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.emognitionInterface import emognitionInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.dataLoaderPyTorch import pytorchDataInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.amigosInterface import amigosInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.dapperInterface import dapperInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.wesadInterface import wesadInterface
from ...dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.caseInterface import caseInterface
from ..featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames  # Functions to extract feature names
from ..modelControl.modelSpecifications.compileModelInfo import compileModelInfo  # Functions with model information
from .compileModelDataHelpers import compileModelDataHelpers


class compileModelData(compileModelDataHelpers):

    def __init__(self, submodel, userInputParams, useTherapyData, accelerator=None):
        super().__init__(submodel, userInputParams, accelerator)
        # Initialize relevant information.
        self.compileModelInfo = compileModelInfo()  # Initialize the model information class.

        # Get the data folder.
        self.trainingFolder = os.path.dirname(__file__) + "/../../../" + self.compileModelInfo.getTrainingDataFolder(useTherapyData=useTherapyData)
        self.fullAnalysisSuffix = '_fullAnalysisParams'

        # Initialize the metadata interfaces.
        self.metaProtocolMap = {
            modelConstants.emognitionDatasetName: emognitionInterface(),
            modelConstants.amigosDatasetName: amigosInterface(),
            modelConstants.dapperDatasetName: dapperInterface(),
            modelConstants.wesadDatasetName: wesadInterface(),
            modelConstants.caseDatasetName: caseInterface()
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

        # Survey information.
        numQuestionOptions = self.compileModelInfo.numQuestionOptions

        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        globalMetaAnalysisClass = globalMetaAnalysis(self.trainingFolder, surveyQuestions=[])
        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = globalMetaAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData=False, metaTraining=False)

        # Compile experimental information.
        userNames = np.unique([i.split(" ")[-2].lower() for i in subjectOrder])
        subjectOrder = np.asarray([np.where(userName.split(" ")[-2].lower() == userNames)[0][0] for userName in subjectOrder])
        activityNames, activityLabels = self.compileModelInfo.extractActivityInformation(experimentalOrder, distinguishBaselines=False)

        # Compile the project data together
        data_to_store = {f"{compiledModelName}": [allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions]}
        data_to_store_Full = {f"{compiledModelName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, experimentalOrder,
                                                       allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes, activityNames, activityLabels, featureNames, numQuestionOptions]}
        self.saveCompiledInfo(data_to_store_Full, compiledModelName + self.fullAnalysisSuffix)
        self.saveCompiledInfo(data_to_store, compiledModelName)

        return allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions

    def compileMetaAnalyses(self, metaDatasetNames, loadCompiledData=False, compiledModelName="compiledMetaTrainingInfo"):
        # Prepare to compile all the metadata analyses.
        metaSurveyQuestions, metaSurveyAnswersList, metaNumQuestionOptions, metaSubjectOrder, metaSurveyAnswerTimes = [], [], [], [], []
        metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaFeatureNames, metaActivityNames, metaActivityLabels = [], [], [], [], []

        plotTrainingData = False
        # For each meta-analysis protocol
        for metadatasetName in metaDatasetNames:
            metaAnalysisProtocol = self.metaProtocolMap[metadatasetName]

            # Prepare the data to go through the training interface.
            streamingOrder, biomarkerFeatureOrder, featureAverageWindows, featureNames, biomarkerFeatureNames = metaAnalysisProtocol.compileTrainingInfo()
            compiledModelFinalName = compiledModelName + f"_{metaAnalysisProtocol.datasetName}"
            numQuestionOptions = metaAnalysisProtocol.numQuestionOptions
            datasetName = metaAnalysisProtocol.datasetName

            print(f"Reading in metadata for {datasetName}")
            # Base case: we are loading in data that was already compiled.
            if loadCompiledData and os.path.isfile(f'{self.compiledInfoLocation}{compiledModelFinalName}{self.compiledExtension}'):
                allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions = self.loadCompiledInfo(compiledModelFinalName)
            else:
                # Collected the training data.
                allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, \
                    subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = metaAnalysisProtocol.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)

                # Compile experimental information.
                activityNames, activityLabels = metaAnalysisProtocol.extractExperimentLabels(experimentalOrder)

                # Compile the project data together
                data_to_store = {f"{compiledModelFinalName}": [allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions]}
                data_to_store_Full = {f"{compiledModelFinalName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, experimentalOrder,
                                                                    allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes, activityNames, activityLabels, featureNames, numQuestionOptions]}
                self.saveCompiledInfo(data_to_store_Full, compiledModelFinalName + self.fullAnalysisSuffix)
                self.saveCompiledInfo(data_to_store, compiledModelFinalName)

            # Organize all the metadata analyses.
            metaRawFeatureIntervalTimes.append(allRawFeatureIntervalTimes)
            metaRawFeatureIntervals.append(allRawFeatureIntervals)
            metaNumQuestionOptions.append(numQuestionOptions)
            metaSurveyAnswersList.append(surveyAnswersList)
            metaSurveyAnswerTimes.append(surveyAnswerTimes)
            metaSurveyQuestions.append(surveyQuestions)
            metaActivityLabels.append(activityLabels)
            metaActivityNames.append(activityNames)
            metaFeatureNames.append(featureNames)
            metaSubjectOrder.append(subjectOrder)

        return metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames

    # -------------------- Machine Learning Preparation -------------------- #

    def compileModelsFull(self, metaDatasetNames, submodel, testSplitRatio, datasetNames):
        # Compile the project data and metadata together
        metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames = self.compileMetaAnalyses(metaDatasetNames, loadCompiledData=True)
        allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions = self.compileProjectAnalysis(loadCompiledData=True)

        # Compile the meta-learning modules.
        allMetaModels, allMetadataLoaders = self.compileModels(metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                                                               metaSubjectOrder, metaFeatureNames, metaDatasetNames, submodel, testSplitRatio, metaTraining=True, specificInfo=None, random_state=42)
        # Compile the final modules.
        allModels, allDataLoaders = self.compileModels(metaRawFeatureIntervalTimes=[allRawFeatureIntervalTimes], metaRawFeatureIntervals=[allRawFeatureIntervals], metaSurveyAnswerTimes=[surveyAnswerTimes], metaSurveyAnswersList=[surveyAnswersList],
                                                       metaSurveyQuestions=[surveyQuestions], metaActivityLabels=[activityLabels], metaActivityNames=[activityNames], metaNumQuestionOptions=[numQuestionOptions], metaSubjectOrder=[subjectOrder],
                                                       metaFeatureNames=[featureNames], metaDatasetNames=datasetNames, submodel=submodel, testSplitRatio=testSplitRatio, metaTraining=False, specificInfo=None, random_state=42)

        return allModels, allDataLoaders, allMetaModels, allMetadataLoaders, metaDatasetNames

    def compileModels(self, metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                      metaSubjectOrder, metaFeatureNames, metaDatasetNames, submodel, testSplitRatio, metaTraining, specificInfo=None, random_state=42):
        # Initialize relevant holders.
        allModelPipelines, lossDataHolders, allDataLoaders = [], [], []

        # Specify model parameters
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = self.modelParameters.getModelInfo(submodel, specificInfo)
        print(f"\nSplitting the data into {'meta-' if metaTraining else ''}models:", flush=True)

        # For each meta-information collected.
        for metadataInd in range(len(metaRawFeatureIntervalTimes)):
            allRawFeatureIntervalTimes = metaRawFeatureIntervalTimes[metadataInd].copy()
            allRawFeatureIntervals = metaRawFeatureIntervals[metadataInd].copy()
            numQuestionOptions = metaNumQuestionOptions[metadataInd].copy()
            surveyAnswersList = metaSurveyAnswersList[metadataInd].copy() - 1
            surveyAnswerTimes = metaSurveyAnswerTimes[metadataInd].copy()
            surveyQuestions = metaSurveyQuestions[metadataInd].copy()
            activityLabels = metaActivityLabels[metadataInd].copy()
            activityNames = metaActivityNames[metadataInd].copy()
            featureNames = metaFeatureNames[metadataInd].copy()
            metadatasetName = metaDatasetNames[metadataInd]
            subjectOrder = metaSubjectOrder[metadataInd]
            activityLabelInd = len(surveyQuestions)

            # Assert the assumptions made about the incoming data
            assert surveyAnswersList.min().item() >= -2, "All ratings must be greater than 0 (exception for -2, which is reserved for missing)."
            assert -1 not in surveyAnswersList, print("surveyAnswersList should contain ratings from 0 to n", flush=True)
            # Specify the incoming dimensions.
            # allRawFeatureIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
            # allRawFeatureTimeIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant

            # ---------------------- Data Preparation ---------------------- #

            # Convert the incoming information to torch tensors.
            surveyAnswerTimes = torch.as_tensor(list(itertools.chain.from_iterable(surveyAnswerTimes)))
            activityLabels = torch.as_tensor(activityLabels).reshape(-1, 1)
            surveyAnswersList = torch.as_tensor(surveyAnswersList)
            allSubjectInds = torch.as_tensor(subjectOrder, dtype=torch.int)

            # Add the human activity recognition to the end.
            allFeatureLabels = torch.hstack((surveyAnswersList, activityLabels))
            # allFeatureLabels dimension: batchSize, numQuestions + 1

            # Remove any experiments and signals that are bad.
            allSignalData, allNumSignalPoints = self._padSignalData(allRawFeatureIntervalTimes, allRawFeatureIntervals, surveyAnswerTimes)
            allSignalData, allNumSignalPoints, featureNames = self._preprocessSignals(allSignalData, allNumSignalPoints, featureNames)
            allFeatureLabels, allSingleClassMasks = self._preprocessLabels(allFeatureLabels)
            # allSmallClassIndices dimension: numLabels, batchSize*  →  *if there are no small classes, the dimension is empty
            # allSignalData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
            # allNumSignalPoints dimension: batchSize, numSignals
            # allFeatureLabels dimension: batchSize, numLabels
            # allSubjectInds dimension: batchSize
            # featureNames dimension: numSignals

            # Compile dataset-specific information.
            numExperiments, numSignals, maxSequenceLength, numSignalChannels = allSignalData.size()
            allExperimentalIndices = torch.arange(0, numExperiments)
            numSubjects = max(allSubjectInds) + 1
            numLabels = allFeatureLabels.size(1)

            # Check the validity of the data before proceeding.
            assert len(featureNames) == numSignals, f"The number of feature names must match the number of signals in {metadatasetName}."
            if numExperiments == 0: print(f"No experiments worthwhile of my time and energy in {metadatasetName}"); continue
            if numSignals == 0: print(f"No signals worthwhile of my time and energy in {metadatasetName}"); continue

            # ---------------------- Test/Train Split ---------------------- #

            # Initialize masks for distinguishing between training and testing data.
            currentTrainingMask = torch.zeros_like(allFeatureLabels, dtype=torch.bool)
            currentTestingMask = torch.zeros_like(allFeatureLabels, dtype=torch.bool)

            # For each type of label/emotion recorded.
            for labelTypeInd in range(numLabels):
                currentFeatureLabels = allFeatureLabels[:, labelTypeInd]
                smallClassMask = allSingleClassMasks[labelTypeInd]
                # smallClassIndices dimension: numSmallClassIndices → containing their indices in the batch.
                # currentFeatureLabels dimension: batchSize

                # Apply the mask to get the valid class data.
                validLabelMask = ~torch.isnan(currentFeatureLabels) & ~smallClassMask  # Dim: batchSize
                currentIndices = allExperimentalIndices[validLabelMask]  # Dim: numValidLabels
                stratifyBy = currentFeatureLabels[validLabelMask]  # Dim: numValidLabels

                # You must have at least two labels per class.
                if len(stratifyBy) == 0 or testSplitRatio < len(torch.unique(stratifyBy)) / len(stratifyBy):
                    print(f"\t\tThe unique label ratio is {len(torch.unique(stratifyBy)) / len(stratifyBy)}. Not training on {surveyQuestions[labelTypeInd]} labels", flush=True)
                    continue

                # Randomly split the data and labels, keeping a balance between testing/training.
                Training_Indices, Testing_Indices = train_test_split(currentIndices.cpu().numpy(), test_size=testSplitRatio, shuffle=True, stratify=stratifyBy.cpu().numpy(), random_state=random_state)

                # Populate the training and testing mask.
                currentTestingMask[Testing_Indices, labelTypeInd] = True
                currentTrainingMask[Training_Indices, labelTypeInd] = True
                # currentTrainingMask dimension: batchSize, numLabels

                # Add back the single class values to the training mask.
                currentTrainingMask[smallClassMask, labelTypeInd] = True
                # currentTestingMask dimension: batchSize, numLabels

                # Assert the validity of the split.
                assert torch.all(~(currentTestingMask & currentTrainingMask)), "Each data point should be in either training, testing, or none, with no overlap."
                # NOTE: The data can be in the training mask or the testing mask or neither; however, it cannot be in both.

            # ---------------------- Data Adjustments ---------------------- #

            # Remove any unused activity labels/names.
            activityNames, allFeatureLabels = self.organizeActivityLabels(activityNames, allFeatureLabels, activityLabelInd)  # Expects inputs/outputs from 0 to n-1

            # Add the demographic information.
            allSignalData = self.addContextualInfo(allSignalData, allNumSignalPoints, allSubjectInds, metadataInd)
            # allSignalData: A torch array of size (batchSize, numSignals, fullDataLength, [timeChannel, signalChannel])

            # ---------------------- Create the Model ---------------------- #

            # Organize the training data into the expected pytorch format.
            batch_size = self.modelParameters.getTrainingBatchSize(submodel, numExperiments=len(allSignalData))
            pytorchDataClass = pytorchDataInterface(batch_size=batch_size, num_workers=0, shuffle=True, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(allSignalData, allFeatureLabels, currentTrainingMask, currentTestingMask)
            reconstructionIndex = emotionDataInterface.getReconstructionIndex(currentTrainingMask)

            # Initialize and train the model class.
            modelPipeline = emotionPipeline(accelerator=self.accelerator, datasetName=metadatasetName, allEmotionClasses=numQuestionOptions, numSubjects=numSubjects,
                                            userInputParams=self.userInputParams, emotionNames=surveyQuestions, activityNames=activityNames, featureNames=featureNames,
                                            submodel=submodel, numExperiments=len(allSignalData), reconstructionIndex=reconstructionIndex)
            modelDataLoader = modelPipeline.acceleratorInterface(modelDataLoader)  # Hugging face integration.

            # Store the information.
            allModelPipelines.append(modelPipeline)
            allDataLoaders.append(modelDataLoader)

            # Gather the number of bits of information per second.
            numGoodEmotions = torch.any(currentTestingMask, dim=0).sum().item()
            print(f"\t{metadatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numLabels - 1}) emotions across {numExperiments} experiments "
                  f"for {numSignals} signals with {round(numExperiments/batch_size, 3)} batches of {batch_size} experiments", flush=True)

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=True)

        return allModelPipelines, allDataLoaders

    def onlyPreloadModelAttributes(self, modelName, datasetNames, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, allDummyModelPipelines=()):
        # Initialize relevant holders.
        userInputParams = {'deviceListed': "cpu", 'submodel': 'emotionPrediction', 'encodedSamplingFreq': 2, 'numSigEncodingLayers': 4, 'numSigLiftedChannels': 4,
                           'compressionFactor': 1.5, 'expansionFactor': 1.5, 'numInterpreterHeads': 4, 'numBasicEmotions': 8, 'finalDistributionLength': 240}

        # Overwrite the location of the saved models.
        self.modelMigration.replaceFinalModelFolder("_finalModels/storedModels/")

        if len(allDummyModelPipelines) == 0:
            print(f"\nInitializing the models for {modelName}:", flush=True)
            # For each model we want.
            for metadataInd in range(len(datasetNames)):
                datasetName = datasetNames[metadataInd]

                # Initialize and train the model class.
                dummyModelPipeline = emotionPipeline(accelerator=self.accelerator, datasetName=datasetName, allEmotionClasses=[], numSubjects=1,
                                                     userInputParams=userInputParams, emotionNames=[], activityNames=[], featureNames=[],
                                                     submodel=loadSubmodel, numExperiments=1, reconstructionIndex=1)
                # Hugging face integration.
                dummyModelPipeline.acceleratorInterface()

                # Store the information.
                allDummyModelPipelines.append(dummyModelPipeline)

        # Load in the model attributes.
        self.modelMigration.loadModels(allDummyModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=False)

        return allDummyModelPipelines
