import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.amigosInterface import amigosInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.caseInterface import caseInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.dapperInterface import dapperInterface
# Import interfaces for the metadata
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.emognitionInterface import emognitionInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.wesadInterface import wesadInterface
from .compileModelDataHelpers import compileModelDataHelpers
from ..featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames  # Functions to extract feature names
from helperFiles.machineLearning.modelControl.Models.pyTorch.dataLoaderPyTorch import pytorchDataInterface
# Import files for training and testing the model
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionPipeline import emotionPipeline
from ..modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ..modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from ..modelControl.modelSpecifications.compileModelInfo import compileModelInfo  # Functions with model information
# Import interfaces for the model's data
from ...dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis


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
        numQuestionOptions = [5] * 10  # PANAS Survey
        numQuestionOptions.extend([4] * 20)  # STAI Survey

        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        globalMetaAnalysisClass = globalMetaAnalysis(self.trainingFolder, surveyQuestions=[])
        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = globalMetaAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData=False, metaTraining=False)

        # Compile experimental information.
        userNames = np.unique([i.split(" ")[-2].lower() for i in subjectOrder])
        subjectOrder = np.asarray([np.where(userName.split(" ")[-2].lower() == userNames)[0][0] for userName in subjectOrder])
        activityNames, activityLabels = self.compileModelInfo.extractActivityInformation(experimentalOrder, distinguishBaselines=False)

        # Compile the project data together
        data_to_store = {f"{compiledModelName}": [allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions]}
        data_to_store_Full = {f"{compiledModelName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, subjectOrder, experimentalOrder,
                                                       allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes, activityNames, activityLabels, featureNames, numQuestionOptions]}
        self.saveCompiledInfo(data_to_store_Full, compiledModelName + self.fullAnalysisSuffix)
        self.saveCompiledInfo(data_to_store, compiledModelName)

        return allRawFeatureIntervalTimes, allCompiledFeatureIntervals, surveyAnswersList, surveyQuestions, activityLabels, activityNames, numQuestionOptions, subjectOrder, featureNames

    def compileMetaAnalyses(self, metaDatasetNames, loadCompiledData=False, compiledModelName="compiledMetaTrainingInfo"):
        # Prepare to compile all the metadata analyses.
        metaSurveyQuestions, metaSurveyAnswersList, metaNumQuestionOptions, metaSubjectOrder, metaSurveyAnswerTimes = [], [], [], [], []
        metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaFeatureNames, metaActivityNames, metaActivityLabels = [], [], [], [], []

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
                allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions = self.loadCompiledInfo(compiledModelFinalName)
            else:
                # Collected the training data.
                allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, \
                    subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = metaAnalysisProtocol.trainingProtocolInterface(streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)

                # Compile experimental information.
                activityNames, activityLabels = metaAnalysisProtocol.extractExperimentLabels(experimentalOrder)

                # Compile the project data together
                data_to_store = {f"{compiledModelName}": [allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions]}
                data_to_store_Full = {f"{compiledModelName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervals, subjectOrder, experimentalOrder,
                                                               allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes, activityNames, activityLabels, featureNames, numQuestionOptions]}
                self.saveCompiledInfo(data_to_store_Full, compiledModelFinalName + self.fullAnalysisSuffix)
                self.saveCompiledInfo(data_to_store, compiledModelFinalName)

            # Organize all the metadata analyses.
            metaCompiledFeatureIntervals.append(allCompiledFeatureIntervals)
            metaRawFeatureTimeIntervals.append(allRawFeatureIntervalTimes)
            metaNumQuestionOptions.append(numQuestionOptions)
            metaSurveyAnswersList.append(surveyAnswersList)
            metaSurveyAnswerTimes.append(surveyAnswerTimes)
            metaSurveyQuestions.append(surveyQuestions)
            metaActivityLabels.append(activityLabels)
            metaActivityNames.append(activityNames)
            metaFeatureNames.append(featureNames)
            metaSubjectOrder.append(subjectOrder)

        return metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames

    # -------------------- Machine Learning Preparation -------------------- #

    def compileModelsFull(self, metaDatasetNames, modelName, submodel, testSplitRatio, datasetNames):
        # Compile the project data and metadata together
        metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames = self.compileMetaAnalyses(metaDatasetNames, loadCompiledData=True)
        allRawFeatureIntervalTimes, allCompiledFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions = self.compileProjectAnalysis(loadCompiledData=True)

        # Compile the meta-learning modules.
        allMetaModels, allMetadataLoaders = self.compileModels(metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                                                               metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, metaTraining=True, specificInfo=None, random_state=42)
        # Compile the final modules.
        allModels, allDataLoaders = self.compileModels(metaRawFeatureTimeIntervals=[allRawFeatureIntervalTimes], metaCompiledFeatureIntervals=[allCompiledFeatureIntervals], metaSurveyAnswerTimes=[surveyAnswerTimes], metaSurveyAnswersList=[surveyAnswersList], metaSurveyQuestions=[surveyQuestions],
                                                       metaActivityLabels=[activityLabels], metaActivityNames=[activityNames], metaNumQuestionOptions=[numQuestionOptions], metaSubjectOrder=[subjectOrder], metaFeatureNames=[featureNames], metaDatasetNames=datasetNames,
                                                       modelName=modelName, submodel=submodel, testSplitRatio=testSplitRatio, metaTraining=False, specificInfo=None, random_state=42)

        return allModels, allDataLoaders, allMetaModels, allMetadataLoaders, metaDatasetNames

    def compileModels(self, metaRawFeatureTimeIntervals, metaCompiledFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                      metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, metaTraining, specificInfo=None, random_state=42):
        # Initialize relevant holders.
        allModelPipelines, lossDataHolders, allDataLoaders = [], [], []

        # Specify model parameters
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = self.modelParameters.getModelInfo(submodel, specificInfo)
        print(f"\nSplitting the data into {'meta-' if metaTraining else ''}models:", flush=True)

        # For each meta-information collected.
        for metadataInd in range(len(metaRawFeatureTimeIntervals)):
            allRawFeatureTimeIntervals = metaRawFeatureTimeIntervals[metadataInd].copy()
            allCompiledFeatureIntervals = metaCompiledFeatureIntervals[metadataInd].copy()
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
            # allCompiledFeatureIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
            # allRawFeatureTimeIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant

            # ---------------------- Data Preparation ---------------------- #

            # Convert the incoming information to torch tensors.
            activityLabels = torch.as_tensor(activityLabels, dtype=torch.float32).reshape(-1, 1)
            surveyAnswersList = torch.as_tensor(surveyAnswersList, dtype=torch.float32)
            subjectOrder = torch.as_tensor(subjectOrder, dtype=torch.int)

            # Add the human activity recognition to the end.
            surveyAnswersList = torch.hstack((surveyAnswersList, activityLabels))
            # surveyAnswersList dimension: batchSize, numQuestions + 1

            # Remove any experiments and signals that are bad.
            allSignalData, allNumSignalPoints = self._padSignalData(allRawFeatureTimeIntervals, allCompiledFeatureIntervals, surveyAnswerTimes)
            allSignalData, allNumSignalPoints, allFeatureLabels, allSubjectInds = self._removeBadExperiments(allSignalData, allNumSignalPoints, surveyAnswersList, subjectOrder)
            allSignalData, allNumSignalPoints, featureNames = self._preprocessSignals(allSignalData, allNumSignalPoints, featureNames)
            allFeatureLabels, allSmallClassIndices = self.organizeLabels(allFeatureLabels, metaTraining)
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
            assert len(featureNames) == numSignals, "The number of feature names must match the number of signals."
            if numExperiments == 0: continue
            if numSignals == 0: continue

            # Gather the number of bits of information per second.
            allSignalTimes = emotionDataInterface.getChannelData(allSignalData, channelName=modelConstants.timeChannel)
            allDataFrequencies = allNumSignalPoints / allSignalTimes[:, :, 0]
            maxExperimentalBits = round(allDataFrequencies.sum(dim=-1).max().item(), 2)
            maxSignalBits = round(allDataFrequencies.max().item(), 2)

            # Print the data information.
            numGoodEmotions = torch.sum(~torch.all(torch.isnan(allFeatureLabels), dim=0)).item()
            print(f"\t{metadatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numLabels - 1}) well-labeled emotions across {numExperiments} experiments "
                  f"with {numSignals} signals with a maximum of {maxExperimentalBits}/experiment and {maxSignalBits}/signal bits of information/second.", flush=True)

            # ---------------------- Test/Train Split ---------------------- #

            # Initialize masks for distinguishing between training and testing data.
            currentTrainingMask = torch.zeros_like(allFeatureLabels, dtype=torch.bool)
            currentTestingMask = torch.zeros_like(allFeatureLabels, dtype=torch.bool)

            # For each type of label/emotion recorded.
            for labelTypeInd in range(numLabels):
                currentFeatureLabels = allFeatureLabels[:, labelTypeInd].clone()
                smallClassIndices = allSmallClassIndices[labelTypeInd]
                # smallClassIndices dimension: numSmallClassIndices → containing their indices in the batch.
                # currentFeatureLabels dimension: batchSize

                # Temporarily remove the single classes.
                currentFeatureLabels[smallClassIndices] = self.missingLabelValue
                validLabelMask = ~torch.isnan(currentFeatureLabels)
                # validLabelMask dimension: batchSize

                # Apply the mask to get the valid class data.
                currentIndices = allExperimentalIndices[validLabelMask]  # Dim: numValidLabels
                stratifyBy = currentFeatureLabels[validLabelMask]  # Dim: numValidLabels

                # You must have at least two labels per class.
                if testSplitRatio < len(torch.unique(stratifyBy)) / len(stratifyBy):
                    print(f"\t\tThe labels do not have enough examples for splitting. Not training on {surveyQuestions[labelTypeInd]} labels", flush=True)
                    continue

                # Randomly split the data and labels, keeping a balance between testing/training.
                Training_Indices, Testing_Indices = train_test_split(currentIndices.cpu().numpy(), test_size=testSplitRatio, shuffle=True, stratify=stratifyBy.cpu().numpy(), random_state=random_state)

                # Populate the training and testing mask.
                currentTestingMask[Testing_Indices, labelTypeInd] = True
                currentTrainingMask[Training_Indices, labelTypeInd] = True

                # Add back the single class values to the training mask.
                currentTrainingMask[smallClassIndices, labelTypeInd] = True

            # ---------------------- Data Adjustments ---------------------- #

            # Remove any unused activity labels/names.
            goodActivityMask = (currentTestingMask[:, activityLabelInd]) | (currentTrainingMask[:, activityLabelInd])
            activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask] \
                = self.organizeActivityLabels(activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask])  # Expects inputs/outputs from 0 to n-1
            allFeatureLabels[~goodActivityMask] = self.missingLabelValue  # Remove any unused activity indices (as the good indices were rehashed)

            # Add the demographic information.
            allSignalData = self.addContextualInfo(allSignalData, allNumSignalPoints, allSubjectInds, metadataInd)
            # allSignalData: A torch array of size (batchSize, numSignals, fullDataLength, [timeChannel, signalChannel])

            # ---------------------- Create the Model ---------------------- #

            # Get the model parameters
            batch_size = self.modelParameters.getTrainingBatchSize(submodel, numExperiments=len(allSignalData))

            # Organize the training data into the expected pytorch format.
            pytorchDataClass = pytorchDataInterface(batch_size=batch_size, num_workers=0, shuffle=True, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(allSignalData, allFeatureLabels, currentTrainingMask, currentTestingMask)

            # Initialize and train the model class.
            modelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=metadatasetName, modelName=modelName, allEmotionClasses=numQuestionOptions,
                                            maxNumSignals=numSignals, numSubjects=numSubjects, userInputParams=self.userInputParams, emotionNames=surveyQuestions, activityNames=activityNames,
                                            featureNames=featureNames, submodel=submodel, debuggingResults=False)

            # Hugging face integration.
            trainingInformation = modelPipeline.getDistributedModels(model=None, submodel=modelConstants.trainingInformation)
            modelDataLoader = modelPipeline.acceleratorInterface(modelDataLoader)
            trainingInformation.addSubmodel(submodel)

            # Store the information.
            allModelPipelines.append(modelPipeline)
            allDataLoaders.append(modelDataLoader)

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
                dummyModelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=datasetName, modelName=modelName, allEmotionClasses=[],
                                                     maxNumSignals=500, numSubjects=1, userInputParams=userInputParams, emotionNames=[], activityNames=[], featureNames=[],
                                                     submodel=loadSubmodel, debuggingResults=True)
                # Hugging face integration.
                dummyModelPipeline.acceleratorInterface()

                # Store the information.
                allDummyModelPipelines.append(dummyModelPipeline)

        # Load in the model attributes.
        self.modelMigration.loadModels(allDummyModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=False)

        return allDummyModelPipelines
