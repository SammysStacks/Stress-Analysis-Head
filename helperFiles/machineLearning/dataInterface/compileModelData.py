from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import torch
import os

from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis
from helperFiles.machineLearning.dataInterface.compileModelDataHelpers import compileModelDataHelpers
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionPipeline import emotionPipeline
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.emognitionInterface import emognitionInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.dataLoaderPyTorch import pytorchDataInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.amigosInterface import amigosInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.dapperInterface import dapperInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.wesadInterface import wesadInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.caseInterface import caseInterface
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo


class compileModelData(compileModelDataHelpers):

    def __init__(self, useTherapyData, accelerator=None, validationRun=False):
        super().__init__(accelerator, validationRun)
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

    def compileModelsFull(self, metaDatasetNames, submodel, testSplitRatio, datasetNames, loadSubmodelDate):
        # Compile the project data and metadata together
        metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions, metaSubjectOrder, metaFeatureNames, metaDatasetNames = self.compileMetaAnalyses(metaDatasetNames, loadCompiledData=True)
        allRawFeatureIntervalTimes, allRawFeatureIntervals, subjectOrder, featureNames, surveyQuestions, surveyAnswerTimes, surveyAnswersList, activityNames, activityLabels, numQuestionOptions = self.compileProjectAnalysis(loadCompiledData=True)

        # Compile the meta-learning modules.
        allMetaModels, allMetadataLoaders = self.compileModels(metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                                                               metaSubjectOrder, metaFeatureNames, metaDatasetNames, submodel=submodel, loadSubmodelDate=loadSubmodelDate, testSplitRatio=testSplitRatio, random_state=42)
        # Compile the final modules.
        allModels, allDataLoaders = self.compileModels(metaRawFeatureIntervalTimes=[allRawFeatureIntervalTimes], metaRawFeatureIntervals=[allRawFeatureIntervals], metaSurveyAnswerTimes=[surveyAnswerTimes], metaSurveyAnswersList=[surveyAnswersList],
                                                       metaSurveyQuestions=[surveyQuestions], metaActivityLabels=[activityLabels], metaActivityNames=[activityNames], metaNumQuestionOptions=[numQuestionOptions], metaSubjectOrder=[subjectOrder],
                                                       metaFeatureNames=[featureNames], metaDatasetNames=datasetNames, submodel=submodel, loadSubmodelDate=loadSubmodelDate, testSplitRatio=testSplitRatio, random_state=42)

        return allModels, allDataLoaders, allMetaModels, allMetadataLoaders, metaDatasetNames

    def compileModels(self, metaRawFeatureIntervalTimes, metaRawFeatureIntervals, metaSurveyAnswerTimes, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                      metaSubjectOrder, metaFeatureNames, metaDatasetNames, submodel, loadSubmodelDate, testSplitRatio, random_state=42):
        # Initialize relevant holders.
        print(f"\nSplitting the data into meta-models.")
        allModelPipelines, lossDataHolders, allDataLoaders = [], [], []

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
            assert surveyAnswersList.min() >= -2, "All ratings must be greater than 0 (exception for -2, which is reserved for missing)."
            assert -1 not in surveyAnswersList, print("surveyAnswersList should contain ratings from 0 to n")
            # Specify the incoming dimensions.
            # allRawFeatureIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*, numBiomarkerFeatures*  ->  *finalDistributionLength, *numBiomarkerFeatures are not constant
            # allRawFeatureTimeIntervals dimension: batchSize, numBiomarkers, finalDistributionLength*  ->  *finalDistributionLength is not constant

            # ---------------------- Data Preparation ---------------------- #

            # Convert the incoming information to torch tensors.
            surveyAnswerTimes = torch.as_tensor(list(itertools.chain.from_iterable(surveyAnswerTimes)))
            activityLabels = torch.as_tensor(activityLabels).reshape(-1, 1)
            allSubjectInds = torch.as_tensor(subjectOrder, dtype=torch.int)
            surveyAnswersList = torch.as_tensor(surveyAnswersList)

            # Add the human activity recognition to the end.
            allFeatureLabels = torch.hstack((surveyAnswersList, activityLabels))
            # allFeatureLabels dimension: batchSize, numQuestions + 1

            # Remove any experiments and signals that are bad.
            allSignalData = self._padSignalData(allRawFeatureIntervalTimes, allRawFeatureIntervals, surveyAnswerTimes)
            allSignalData, featureNames = self._preprocessSignals(allSignalData, featureNames, metadatasetName)
            allFeatureLabels, allSingleClassMasks = self._preprocessLabels(allFeatureLabels)
            # allSmallClassIndices dimension: numLabels, batchSize* → *if there are no small classes, the dimension is empty
            # allSignalData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
            # allNumSignalPoints dimension: batchSize, numSignals
            # allFeatureLabels dimension: batchSize, numLabels
            # allSubjectInds dimension: batchSize
            # featureNames dimension: numSignals

            # Compile the valid indices.
            validDataMask = emotionDataInterface.getValidDataMask(allSignalData)  # Dim: batchSize, numSignals, maxSequenceLength
            allSmallClassesMask = validDataMask.sum(dim=-1) <= self.minSignalPresentCount  # Dim: batchSize, numSignals
            validSignalInds = 0 < validDataMask.sum(dim=-1)  # Dim: batchSize, numSignals

            # Get dimensional information.
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
            currentTrainingMask = torch.zeros((numExperiments, numLabels + numSignals), dtype=torch.bool)
            currentTestingMask = torch.zeros((numExperiments, numLabels + numSignals), dtype=torch.bool)

            # For each type of label/emotion recorded.
            for labelTypeInd in range(numLabels + numSignals):
                if labelTypeInd < numLabels:  # Emotion labels
                    currentFeatureLabels = allFeatureLabels[:, labelTypeInd]  # Dim: batchSize
                    smallClassMask = allSingleClassMasks[labelTypeInd]  # Dim: batchSize

                    # Apply the mask to get the valid class data.
                    validLabelMask = ~torch.isnan(currentFeatureLabels) & ~smallClassMask  # Dim: batchSize
                    currentClassLabels = currentFeatureLabels[validLabelMask].int()  # Dim: numValidLabels
                    currentIndices = allExperimentalIndices[validLabelMask]  # Dim: numValidLabels
                    stratifyBy = currentClassLabels.detach().cpu().numpy()  # Convert to numpy.
                else:
                    signalInd = labelTypeInd - numLabels
                    validLabelMask = validSignalInds[:, signalInd]  # Dim: batchSize
                    smallClassMask = allSmallClassesMask[:, signalInd]  # Dim: batchSize

                    # Apply the mask to get the valid class data.
                    currentIndices = allExperimentalIndices[validLabelMask & ~smallClassMask].int()  # Dim: numValidLabels
                    stratifyBy = None

                # Randomly split the data and labels, keeping a balance between testing/training.
                Training_Indices, Testing_Indices = train_test_split(currentIndices.detach().cpu().numpy(), test_size=testSplitRatio, shuffle=True, stratify=stratifyBy, random_state=random_state)

                # Populate the training and testing mask.
                currentTestingMask[Testing_Indices, labelTypeInd] = True
                currentTrainingMask[Training_Indices, labelTypeInd] = True
                # currentTrainingMask dimension: batchSize, numLabels

                # Add back the single class values to the training mask.
                if smallClassMask is not None: currentTrainingMask[smallClassMask, labelTypeInd] = True
                # currentTestingMask dimension: batchSize, numLabels

                # Assert the validity of the split.
                assert torch.all(~(currentTestingMask & currentTrainingMask)), "Each data point should be in either training, testing, or none, with no overlap."
                # NOTE: The data can be in the training mask or the testing mask or neither; however, it cannot be in both.

            # ---------------------- Data Adjustments ---------------------- #

            # Remove any unused activity labels/names and add the contextual information.
            activityNames, allFeatureLabels = self.organizeActivityLabels(activityNames, allFeatureLabels, activityLabelInd)  # Expects inputs/outputs from 0 to n-1
            allSignalData = self.addContextualInfo(allSignalData, allSubjectInds, metadataInd)
            # allSignalData: batchSize, numSignals, fullDataLength, [timeChannel, signalChannel]

            # ---------------------- Create the Model ---------------------- #

            # Organize the training data into the expected pytorch format.
            batch_size = self.modelParameters.getTrainingBatchSize(submodel, numExperiments=len(allSignalData), datasetName=metadatasetName)
            pytorchDataClass = pytorchDataInterface(batch_size=batch_size, num_workers=0, shuffle=True, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(allSignalData, allFeatureLabels, currentTrainingMask, currentTestingMask)

            # Initialize and train the model class.
            modelPipeline = emotionPipeline(accelerator=self.accelerator, datasetName=metadatasetName, allEmotionClasses=numQuestionOptions, numSubjects=numSubjects,
                                            emotionNames=surveyQuestions, activityNames=activityNames, featureNames=featureNames, submodel=submodel, numExperiments=len(allSignalData))
            modelDataLoader = modelPipeline.acceleratorInterface(modelDataLoader)  # Hugging face integration.

            # Set the emotion class weights.
            modelPipeline.assignClassWeights(allFeatureLabels, currentTrainingMask, currentTestingMask)

            # Store the information.
            allModelPipelines.append(modelPipeline)
            allDataLoaders.append(modelDataLoader)

            # Gather the number of bits of information per second.
            numGoodEmotions = torch.any(currentTestingMask, dim=0).sum().item() - numSignals
            print(f"\t{metadatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numLabels - 1}) emotions across {numExperiments} experiments "
                  f"for {numSignals} signals with {round(numExperiments/batch_size, 3)} batches of {batch_size} experiments")

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allModelPipelines, submodel, loadSubmodelDate, loadSubmodelEpochs=-1, loadModelAttributes=True, loadModelWeights=True)

        return allModelPipelines, allDataLoaders
