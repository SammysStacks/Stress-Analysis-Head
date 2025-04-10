import os
import itertools
import concurrent.futures

from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.helperMethods.filteringProtocols import filteringMethods
from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.helperMethods.universalProtocols import universalMethods
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing.excelFormatting import handlingExcelFormat
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing.saveDataProtocols import saveExcelData
from helperFiles.dataAcquisitionAndAnalysis.streamingProtocols import streamingProtocols
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.featureAnalysis.featurePlotting import featurePlotting
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.trainingProtocols import trainingProtocols


class globalMetaAnalysis(handlingExcelFormat):

    def __init__(self, trainingFolder, surveyQuestions):
        super().__init__()
        self.trainingFolder = trainingFolder
        self.surveyQuestions = surveyQuestions
        self.savedFeatureFolder = self.trainingFolder + self.saveFeatureFolder
        self.deviceType = "serial"

        # Define general classes to process data.
        self.saveInputs = saveExcelData()
        self.filteringMethods = filteringMethods()
        self.universalMethods = universalMethods()
        self.compileFeatureNames = compileFeatureNames()
        self.analyzeFeatures = featurePlotting(self.trainingFolder + "dataAnalysis/", overwrite=False)

    # ------------------ Interface with Training Protocols ----------------- #

    def extractFeatures(self, allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                        streamingOrder, biomarkerFeatureOrder, featureAverageWindows, filteringOrders, metadatasetName, reanalyzeData=False, showPlots=True, analyzeSequentially=False):
        # Prepare the data for each subject for parallel processing
        subjects_data = [
            (
                subjectOrder[i],
                allCompiledDatas[i],
                allExperimentalTimes[i],
                allExperimentalNames[i],
                allSurveyAnswerTimes[i],
                allSurveyAnswersList[i],
                allContextualInfo[i],
                streamingOrder,
                biomarkerFeatureOrder,
                featureAverageWindows,
                filteringOrders,
                metadatasetName,
                reanalyzeData,
                showPlots
            ) for i in range(len(subjectOrder))
        ]

        # Analyze the data sequentially for plotting large datasets.
        if showPlots and metadatasetName in [modelConstants.dapperDatasetName, modelConstants.caseDatasetName]:
            print("\tAnalyzing sequentially")
            analyzeSequentially = True

        if analyzeSequentially:
            for i in subjects_data:
                self.processFeatures(i)
        else:
            # Use ProcessPoolExecutor to process each subject in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(self.processFeatures, subjects_data)

    def processFeatures(self, subject_data):
        # Unpack data
        (
            subjectName,
            compiledData_eachFreq,
            experimentTimes,
            experimentNames,
            currentSurveyAnswerTimes,
            currentSurveyAnswersList,
            contextualInfo,
            streamingOrder,
            biomarkerFeatureOrder,
            featureAverageWindows,
            filteringOrders,
            interfaceType,
            reanalyzeData,
            showPlots
        ) = subject_data

        # Organize the data for streaming.
        allBiomarkerFeatureOrders = []
        allFeatureAverageWindows = []
        biomarkerFeatureNames = []
        allStreamingOrders = []
        allFilteringOrders = []
        allFeaturesExtracting = []
        signalPointer = 0

        # Recompile considering the segmentation of the features.
        for segmentationInd in range(len(compiledData_eachFreq)):
            numSignals = len(compiledData_eachFreq[segmentationInd][1])

            # Separate out the relevant signals.
            currentBiomarkerFeatureOrder = biomarkerFeatureOrder[signalPointer:signalPointer + numSignals]
            currentFeatureAverageWindows = featureAverageWindows[signalPointer:signalPointer + numSignals]
            currentFilteringOrders = filteringOrders[signalPointer:signalPointer + numSignals]
            currentStreamingOrder = streamingOrder[signalPointer:signalPointer + numSignals]

            # Organize the biomarker order.
            currentFeatureNames, currentBiomarkerFeatureNames, currentBiomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(currentBiomarkerFeatureOrder)

            # Organize the segmentation block
            allBiomarkerFeatureOrders.append(currentBiomarkerFeatureOrder)
            allFeatureAverageWindows.append(currentFeatureAverageWindows)
            biomarkerFeatureNames.extend(currentBiomarkerFeatureNames)
            allFeaturesExtracting.append(currentBiomarkerFeatureOrder)
            allFilteringOrders.append(currentFilteringOrders)
            allStreamingOrders.append(currentStreamingOrder)

            # Reset for the next round.
            signalPointer += numSignals

        # Stream the data.
        self.streamData(subjectName, allStreamingOrders, allBiomarkerFeatureOrders, allFeatureAverageWindows, allFilteringOrders, compiledData_eachFreq, experimentTimes,
                        experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, contextualInfo, interfaceType, allFeaturesExtracting, biomarkerFeatureNames, reanalyzeData, showPlots=showPlots)

    def streamData(self, subjectName, allStreamingOrders, allBiomarkerFeatureOrders, allFeatureAverageWindows, allFilteringOrders, compiledData_eachFreq, experimentTimes,
                   experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, contextualInfo, interfaceType, allFeaturesExtracting, biomarkerFeatureNames, reanalyzeData=False, showPlots=False):
        # Check if the features have already been extracted.        
        saveFeatureFile = self.savedFeatureFolder + subjectName + self.saveFeatureFile_Appended
        if os.path.exists(saveFeatureFile) and not reanalyzeData:
            print("\tNot reanalyzing file")
            return None

        # Get the flattened versions.
        biomarkerFeatureOrderFull = list(itertools.chain(*allBiomarkerFeatureOrders))

        rawFeatureHolder = []
        rawFeatureTimesHolder = []
        # Organize the data for streaming.
        for compiledDataInd in range(len(compiledData_eachFreq)):
            currentBiomarkerFeatureOrder = allBiomarkerFeatureOrders[compiledDataInd]
            currentFeatureAverageWindows = allFeatureAverageWindows[compiledDataInd]
            currentFeaturesExtracting = allFeaturesExtracting[compiledDataInd]
            currentFilteringOrders = allFilteringOrders[compiledDataInd]
            currentStreamingOrder = allStreamingOrders[compiledDataInd]
            compiledRawData = compiledData_eachFreq[compiledDataInd]

            # Initialize instance to analyze the data
            readData = streamingProtocols(deviceType="serial", mainSerialNum=None, modelClasses=[], actionControl=None, numPointsPerBatch=2048576, moveDataFinger=1048100, streamingOrder=currentStreamingOrder,
                                          extractFeaturesFrom=currentFeaturesExtracting, featureAverageWindows=currentFeatureAverageWindows, voltageRange=(None, None), plotStreamedData=False)
            readData.resetGlobalVariables()
            # Change filter of the analyses.
            for analysisTypeInd in range(len(currentBiomarkerFeatureOrder)):
                analysisType = currentBiomarkerFeatureOrder[analysisTypeInd]
                cutOffFreqs = currentFilteringOrders[analysisTypeInd]

                readData.analysisProtocols[analysisType].cutOffFreq = cutOffFreqs

            # Extract and analyze the raw data.
            readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, self.surveyQuestions, subjectInformationAnswers=[], subjectInformationQuestions=[], filePath=interfaceType + " " + subjectName)
            # Compile all the features from each signal
            rawFeatureHolder.extend(readData.rawFeatureHolder.copy())
            rawFeatureTimesHolder.extend(readData.rawFeatureTimesHolder.copy())

            if showPlots:
                # Plot the signals
                self.analyzeFeatures.overwrite = True
                self.analyzeFeatures.plotRawData(readData, compiledRawData, currentSurveyAnswerTimes, experimentTimes, experimentNames, currentStreamingOrder, folderName=interfaceType + " " + subjectName + "/rawSignals/")

        # Assert the integrity of data combination
        assert len(rawFeatureHolder) == len(biomarkerFeatureOrderFull)
        assert len(rawFeatureTimesHolder) == len(biomarkerFeatureOrderFull)

        # Save the features to be analyzed in the future.
        trainingExcelFile = self.trainingFolder + subjectName + "/" + subjectName + ".xlsx"
        self.saveInputs.saveRawFeatures(rawFeatureTimesHolder, rawFeatureHolder, biomarkerFeatureNames, biomarkerFeatureOrderFull, experimentTimes, experimentNames, currentSurveyAnswerTimes,
                                        currentSurveyAnswersList, self.surveyQuestions, contextualInfo[1], contextualInfo[0], trainingExcelFile)

    def trainingProtocolInterface(self, streamingOrder, biomarkerFeatureOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData=True, metaTraining=True):
        # Initialize instance to train the data
        readData = streamingProtocols(deviceType="serial", mainSerialNum=None, modelClasses=[], actionControl=None, numPointsPerBatch=2048576, moveDataFinger=1048100, streamingOrder=streamingOrder,
                                      extractFeaturesFrom=biomarkerFeatureOrder, featureAverageWindows=featureAverageWindows, voltageRange=(None, None), plotStreamedData=False)
        trainingInterface = trainingProtocols(self.deviceType, biomarkerFeatureNames, streamingOrder, biomarkerFeatureOrder, self.savedFeatureFolder, readData)

        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = trainingInterface.streamTrainingData(featureAverageWindows, plotTrainingData=plotTrainingData, reanalyzeData=False, metaTraining=metaTraining, reverseOrder=False)

        return allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervalTimes, allRawFeatureIntervals, allCompiledFeatureIntervalTimes, allCompiledFeatureIntervals, \
            subjectOrder, experimentalOrder, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes
