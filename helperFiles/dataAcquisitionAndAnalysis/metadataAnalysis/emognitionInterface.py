
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import os
import json
import numpy as np
from datetime import datetime
# Module to Sort Files in Order
from natsort import natsorted
# Modules to plot
import matplotlib.pyplot as plt

# Import excel data interface
from .globalMetaAnalysis import globalMetaAnalysis

# -------------------------------------------------------------------------- #
# ------------------------- Extract Data from Excel ------------------------ #
    
class emognitionInterface(globalMetaAnalysis):
    
    def __init__(self):
        # Specify the metadata file locations.
        self.subjectFolders = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/EMOGNITION/"
        
        # Initialize EMOGNITION survey information.
        self.emotionQuestions = ['AWE', 'DISGUST', 'SURPRISE', 'ANGER', 'ENTHUSIASM', 'LIKING', 'FEAR', 'AMUSEMENT', 'SADNESS'] # Rated 1 to 5
        self.dimQuestions = ['VALENCE', 'AROUSAL', 'MOTIVATION'] # Rated 1 to 9
        # Compile all survey questions.
        self.surveyQuestions = [question.lower() for question in self.emotionQuestions]
        self.surveyQuestions.extend([question.lower() for question in self.dimQuestions])
        # Compile the scoring ranges
        self.numQuestionOptions = [5]*len(self.emotionQuestions)
        self.numQuestionOptions.extend([9]*len(self.dimQuestions))
        # Specify the current dataset.
        self.datasetName = "emognition"
        
        # Initialze the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions) # Intiailize meta analysis.
        
    def getSeconds(self, timestamp):
        timestamp = timestamp.replace(".", ":")
        # Extract the time part from the timestamp
        time_str = timestamp.split("T")[1]
        timeSplits = time_str.split(":")
        numZones = len(timeSplits)
        
        # Find the format of the times
        if numZones == 3:
            time_format = "%H:%M:%S"
        elif numZones == 4:
            time_format = "%H:%M:%S:%f"
        else:
            assert False, timestamp

        # Convert the time part to a datetime object (ignoring date information)
        time_obj = datetime.strptime(time_str, time_format)
        # Get the total seconds from the time object
        total_seconds = (time_obj - time_obj.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        
        return total_seconds
    
    def getData(self, showPlots = False):
        # Initialize data holders.
        allExperimentalTimes = []; allExperimentalNames = []
        allSurveyAnswerTimes = []; allSurveyAnswersList = []
        allCompiledDatas = []; subjectOrder = []
        allContextualInfo = []
        
        # Information about what is in the EMOGNITION dataset.
        experimentTypes = ['BASELINE', 'NEUTRAL', 'AWE', 'DISGUST', 'SURPRISE', 'ANGER', 'ENTHUSIASM', 'LIKING', 'FEAR', 'AMUSEMENT', 'SADNESS']
        experimentSections = ['WASHOUT', 'STIMULUS', 'QUESTIONNAIRES']
        recordingDevices = ['EMPATICA', 'SAMSUNG_WATCH', 'MUSE']
        # Information about missing data
        missingData = np.array([
            ('29', 'MUSE'),
            ('30', 'SAMSUNG_WATCH'),
            ('32', 'MUSE'),
            ('48', 'EMPATICA'),
            ('46', 'TEMP'),
            ('57', 'IDK'),
        ])
        
        # Information about the recorded signals.
        allExpectedSignalTypes = np.array(
            ['BVP', 'TEMP', 'IBI', 'ACC', 'EDA', \
             'acc', 'gyr', 'rot', 'heartRate', 'PPInterval', 'BVPRaw', 'BVPProcessed', \
             # 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10',  # This one is not all of MUSE, but relevant ones.
             ]
        )
        allFinalSignalTypes = allExpectedSignalTypes[[0, 1, 4, 8, 9, 11]]

        # For each subject in the training folder.
        for subjectFolderName in natsorted(os.listdir(self.subjectFolders)):
            subjectFolder = self.subjectFolders + subjectFolderName + "/"
            if subjectFolderName in missingData[:, 0]: continue
                                    
            # Only look at subject folders, while ignoring other analysis folders.
            if os.path.isdir(subjectFolder) and self.universalMethods.isNumber(subjectFolderName):
                print("\nExtracting data from EMOGNITION subject:", subjectFolderName)
                # Specify the files to extract data from                
                questionaireFile = subjectFolder + subjectFolderName + "_QUESTIONNAIRES.json"
                
                # Read the JSON questionaire data file.
                with open(questionaireFile, "r") as json_file:
                    questionaireData = json.load(json_file)
                    
                # Compile contextual information.
                subjectAge = questionaireData['metadata']['age']
                subjectGender = questionaireData['metadata']['gender']
                subjectGender = 0 if subjectGender[0] == 'f' else 1
                subjectContext = [["Age", "Gender"], [subjectAge, subjectGender]]
                
                currentSurveyAnswersList = []
                movieOrder = questionaireData['metadata']['movie_order']
                # For each experiment performed on the subject.
                for experimentInd in range(len(questionaireData['questionnaires'])):
                    experimentQuestionaireData = questionaireData['questionnaires'][experimentInd]
                                        
                    # Get the emotional ratings from the questionaires.
                    emotionData = experimentQuestionaireData['emotions']
                    samData = experimentQuestionaireData['sam']
                    
                    # Organize the emotion data
                    emotionsRecorded = list(emotionData.keys())
                    emotionRatings = list(emotionData.values())
                    # Organize the sam data
                    dimRecorded = list(samData.keys())
                    dimRatings = list(samData.values())
                    
                    # Assert data integrity
                    assert self.dimQuestions == dimRecorded
                    assert self.emotionQuestions == emotionsRecorded
                    
                    # Compile the data together
                    currentSurveyAnswersList.append(emotionRatings)
                    currentSurveyAnswersList[-1].extend(dimRatings)
                # Assert the integrity of information extraction.
                assert len(currentSurveyAnswersList) == len(experimentTypes) 
                assert len(currentSurveyAnswersList) == len(movieOrder) 
                assert len(experimentTypes) == len(movieOrder) 
                
                # Establish data holder for the variables.
                compiledData_eachFreq = []; currentSurveyAnswerTimes = []
                experimentNames = []; experimentTimes = []                
                
                combinedSignalInfos = [[] for _ in range(len(allFinalSignalTypes))]
                allSectionStartTimes = [[] for _ in range(len(allFinalSignalTypes))]
                # For each experiment ran on the subject.
                for experimentTypeInd in range(len(movieOrder)):
                    experimentType = movieOrder[experimentTypeInd]
                    
                    surveyStartTimes = []
                    experimentalStartTimes = []
                    # For each section of the experiment (baseline, stimulus, or questionaire).
                    for experimentSectionInd in range(len(experimentSections)):
                        experimentSection = experimentSections[experimentSectionInd]
                        if experimentType == "BASELINE" and experimentSection == "WASHOUT": continue
                        
                        combinedSignalTypes = []
                        # For each recording device with signasl.
                        for recordingDeviceInd in range(len(recordingDevices)):
                            recordingDevice = recordingDevices[recordingDeviceInd]
                            if recordingDevice == "MUSE": continue                             
                            # Get the corresponding information for this run.
                            dataFile = subjectFolder + subjectFolderName + f"_{experimentType}_{experimentSection}_{recordingDevice}.json"
                                                                                    
                            # Load in data file (compiled by researchers).
                            with open(dataFile, 'r') as json_file:
                                data = json.load(json_file)
                            # Extract the signal information.
                            signalTypes = list(data.keys())
                            allSignalInfos = list(data.values())
                            
                            # MUSE has timepoints in the first column
                            if recordingDevice == "MUSE":   
                                # Extract and convert the time
                                timepoints = np.array(allSignalInfos[0])
                                timepoints = np.vectorize(self.getSeconds)(timepoints)
                                ### TODO: There are negative time differences and duplicate times.
                                                        
                            # For each type of signal.
                            for signalTypeInd in range(len(signalTypes)):
                                singalInfo = np.array(allSignalInfos[signalTypeInd])
                                signalType = signalTypes[signalTypeInd]
                                # Check to see if we are compiling this signal.
                                if signalType not in allFinalSignalTypes:
                                    continue
                                compiledInd = np.where(allFinalSignalTypes == signalType)[0][0]

                                if recordingDevice == "MUSE":  
                                    singalInfo = np.vstack((timepoints, singalInfo)).T
                                else:
                                    # Extract the time and data points.
                                    timepoints = singalInfo[:, 0]
                                    # Convert to timestamps to seconds.
                                    timepoints = np.vectorize(self.getSeconds)(timepoints)
                                    singalInfo[:, 0] = timepoints
                                    
                                # Organize the signals into one experiment.
                                allSectionStartTimes[compiledInd].append(timepoints[0])
                                combinedSignalInfos[compiledInd].extend(singalInfo)
                                combinedSignalTypes.append(signalType)
                                
                                if experimentSection == "STIMULUS":
                                    experimentalStartTimes.append(timepoints[0])
                                elif experimentSection == "QUESTIONNAIRES":
                                    surveyStartTimes.append(timepoints[0])
                        
                        # Assert the integrity of data collection
                        assert len(allFinalSignalTypes) == len(combinedSignalTypes)
                        assert (allFinalSignalTypes == combinedSignalTypes).all()
                        
                    # Convert to numpy arrays
                    surveyStartTime = min(surveyStartTimes)
                    experimentStartTime = max(experimentalStartTimes)
                    
                    # Compile experimental information
                    experimentNames.append(experimentType)
                    experimentTimes.append([experimentStartTime, surveyStartTime])
                    currentSurveyAnswerTimes.append(surveyStartTime)
                
                # Calculate the offset to zero the times.
                allSectionStartTimes = np.array(allSectionStartTimes)
                offsetTime = min(allSectionStartTimes[:, 0])
                # Remove the offset time from the data.
                experimentTimes = np.array(experimentTimes) - offsetTime
                currentSurveyAnswerTimes = np.array(currentSurveyAnswerTimes) - offsetTime
                    
                for signalInd in range(len(combinedSignalInfos)):
                    singalInfo = np.array(combinedSignalInfos[signalInd]).astype(float)
                    # Convert uS to S.
                    if combinedSignalTypes[signalInd] == "EDA":
                        singalInfo[:, 1] = singalInfo[:, 1]*1E-6
                    
                    # Select the time and data
                    timepoints = singalInfo[:, 0] - offsetTime
                    allData = singalInfo[:, 1:].T
                    # Zero out missing values.
                    allData[np.isnan(allData)] = 0 # Should be only for EEG I think.
                    # Compile the final data.
                    compiledData_eachFreq.append([timepoints, allData])
                    
                    if showPlots:
                        plt.rcdefaults()
                        for dataInd in range(len(allData)):
                            plt.plot(timepoints, allData[dataInd])
                        # plt.xlim(500, 510)
                        plt.title(combinedSignalTypes[signalInd])
                        plt.xlabel("Time (Seconds)")
                        plt.ylabel("Arbitrary Units (AU)")
                        plt.show()
                
                # Assert the integrity of data compilation
                assert len(experimentTimes) == len(experimentNames) \
                    == len(currentSurveyAnswerTimes) == len(currentSurveyAnswersList)
                
                # Organize the data collected.
                subjectOrder.append(subjectFolderName)
                allContextualInfo.append(subjectContext)
                allCompiledDatas.append(compiledData_eachFreq)
                allExperimentalTimes.append(experimentTimes)
                allExperimentalNames.append(experimentNames)
                allSurveyAnswerTimes.append(currentSurveyAnswerTimes)
                allSurveyAnswersList.append(currentSurveyAnswersList)
                print("\tFinished data extraction");
                
        # Convert to numpy arrays
        subjectOrder = np.array(subjectOrder)
        allContextualInfo = np.array(allContextualInfo)
        allExperimentalTimes = np.array(allExperimentalTimes)
        allExperimentalNames = np.array(allExperimentalNames)
        allSurveyAnswerTimes = np.array(allSurveyAnswerTimes)
        allSurveyAnswersList = np.array(allSurveyAnswersList)
                
        return allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo
    
    def extractExperimentLabels(self, allExperimentalNames):

        activityNames = np.unique(allExperimentalNames)
        
        activityLabels = []
        for activityName in allExperimentalNames:
            activityLabels.append(np.where((activityNames == activityName))[0][0])

        return activityNames, activityLabels
    
    def getStreamingInfo(self):
        # Feature information
        streamingOrder = ['lowFreq', 'temp', 'eda', 'lowFreq', 'lowFreq', 'lowFreq']
        biomarkerOrder = ['lowFreq', 'temp', 'eda', 'lowFreq', 'lowFreq', 'lowFreq']
        filteringOrders = [[None, 20], [None, 0.1], [None, None], [None, 5], [None, 5], [None, None]]  # Sampling Freq: 64, 4, 4, 10, 10, 20 (Hz); Need 1/2 frequency at max.
        featureAverageWindows = [30, 30, 30, 30, 30, 30] # ["BVP", 'TEMP', 'EDA', 'heartRate', 'PPInterval', 'BVPProcessed']
        
        # Feature information: MUSE
        # streamingOrder.extend(['eeg', 'eeg', 'eeg', 'eeg'])
        # biomarkerOrder.extend(['eeg', 'eeg', 'eeg', 'eeg'])
        # filteringOrders.extend([[None, None], [None, None], [None, None], [None, None]])  # Sampling Freq: 256, 256, 256, 256 (Hz); Need 1/2 frequency at max.
        # featureAverageWindows.extend([30, 30, 30, 30]) # ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        
        return streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders
    
    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = self.getStreamingInfo()
        featureNames, biomarkerFeatureNames, biomarkerOrder = self.compileFeatureNames.extractFeatureNames(biomarkerOrder)
            
        return streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames
        
        
if __name__ == "__main__":
    # Initialize metadata analysis class.
    emognitionAnalysisClass = emognitionInterface()
    
    analyzingData = True
    trainingData = False
    
    if analyzingData:
        # Extract the metadata
        allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, \
            allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo = emognitionAnalysisClass.getData(showPlots = False)
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = emognitionAnalysisClass.getStreamingInfo()
        # Analyze and save the metadata features
        emognitionAnalysisClass.extractFeatures(allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                                                      streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders, interfaceType = 'emognition', reanalyzeData = True, showPlots = True)
    
    if trainingData:
        # Prepare the data to go through the training interface.
        streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames = emognitionAnalysisClass.compileTrainingInfo()
        
        plotTrainingData = False
        # Collected the training data.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
                experimentOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = emognitionAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)
    
    
    
    
    
    