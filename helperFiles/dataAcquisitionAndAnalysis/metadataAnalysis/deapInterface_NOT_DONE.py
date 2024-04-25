
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import os
import pickle
import numpy as np
import pandas as pd
import scipy
# Module to Sort Files in Order
from natsort import natsorted
import re
# Modules to plot
import matplotlib.pyplot as plt

# Import excel data interface
import globalMetaAnalysis

# -------------------------------------------------------------------------- #
# ------------------------- Extract Data from Excel ------------------------ #
    
class deapInterface(_globalMetaAnalysis.globalMetaAnalysis):
    
    def __init__(self):
        self.debug = False
        # Specify the metadata file locations.
        self.subjectFolders = os.path.dirname(__file__) + "/../../../Data/_metaDatasets/DEAP/data/interpolated/physiological/"
        self.subjectFoldersAnnotations = os.path.dirname(__file__) + "/../../../Data/_metaDatasets/DEAP/data/interpolated/annotations/"
        self.subjectFoldersMetaData = os.path.dirname(__file__) + "/../../../Data/_metaDatasets/DEAP/metadata/"
        # Initialize DEAP survey information.
        # Compile all survey questions. NOTE: Ignoring the extra 2 in the stress condition.
        self.surveyQuestions = ["valence", "arousal"]
        self.demographicsQuestions = ['Age', 'Gender']
        # Compile the scoring ranges
        self.numQuestionOptions = [9]*len(self.surveyQuestions)
        # Specify the current dataset.
        self.datasetName = "deap"
                
        # Define DEAP-specific parameters
        self.deap_streamingOrder = ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
        self.deap_streamingConversions = [10**-3, 1, 10**-6, 1, 1, 1, 1, 1]
        
        df_videos = pd.read_excel(self.subjectFoldersMetaData + 'videos.xlsx')
        self.videoIDtoName = {row['Video-ID']: row['Source (Year)'] + ', ' + row['Video-label'] for index, row in df_videos.dropna(subset=['Video-ID']).iterrows()}
        self.activityNames = ['amusing', 'blue screen', 'boring', 'relaxed', 'scary', 'startVid']
        self.trialDuration = 5 * 60 # 5 minute sliding window per trial in experiment
        self.trialShift = 30 # 30 second sliding window shift
        self.videoStartBuffer = 60 # 60 seconds for emotional response from video
        self.surveyAggregation = 0.5 # emotion responses are averaged over last 3 seconds of trial
        # Initialze the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions) # Intiailize meta analysis.
        
        
    def cleanFileData(self, sychronizedData, emotionAnnotations):
        sychronizedData['daqtime'] = sychronizedData['daqtime'] / 1000
        emotionAnnotations['jstime'] = emotionAnnotations['jstime'] / 1000
        
        return sychronizedData, emotionAnnotations
    
    def aggregateSurveysPerTrial(self, surveys):
        # surveys is pandas df only containing the responses per given trial
        return [scipy.stats.trim_mean(surveys[question], 0.1) for question in self.surveyQuestions]
    
    def getData(self, showPlots = False):
        # Initialize data holders.
        allExperimentalTimes = []; allExperimentalNames = []
        allSurveyAnswerTimes = []; allSurveyAnswersList = []
        allSynchronizedData = []; subjectOrder = []; allContextualInfo = []
        
        df_metadata = pd.read_excel(self.subjectFoldersMetaData + 'participants.xlsx')

        # For each subject in the training folder.
        for subjectSignalDataName in natsorted(os.listdir(self.subjectFolders)):
            # Do not alayze hidden files.
            if subjectSignalDataName.startswith((".", "~")): continue
            
            # Only look at subject folders, while ignoring other analysis folders.
            if os.path.exists(self.subjectFolders + subjectSignalDataName) and subjectSignalDataName.endswith('.csv'):
                subjectName = subjectSignalDataName.split(".")[0]
                if subjectName not in ['sub_1']: continue
                print("\nExtracting data from DEAP subject:", subjectName)
                
                # Get Contextual Info -----------------------------------------
                
                subjectInfo = df_metadata.iloc[int(subjectName.split("_")[-1]) - 1]
                allContextualInfo.append([self.demographicsQuestions, [subjectInfo['Age-Group'], subjectInfo['Sex']]])
                # if self.debug:
                print(allContextualInfo)
                # -------------------------------------------------------------
                
                df_subjectSynchronizedData = pd.read_csv(self.subjectFolders + subjectSignalDataName)
                df_subjectEmotionAnnotations = pd.read_csv(self.subjectFoldersAnnotations + subjectSignalDataName)
                
                df_subjectSynchronizedData, df_subjectEmotionAnnotations = self.cleanFileData(df_subjectSynchronizedData, df_subjectEmotionAnnotations)
                
                if self.debug:
                    print(df_subjectSynchronizedData.head())
                    print(df_subjectEmotionAnnotations.head())
                
                unique_videos = set(df_subjectSynchronizedData['video'].unique())
                assert unique_videos == set(df_subjectEmotionAnnotations['video'].unique())
                
                videoBufferTimestamps = {}
                for video in unique_videos:
                    videoStartTime = df_subjectSynchronizedData[df_subjectSynchronizedData['video'] == video].iloc[0]['daqtime']
                    videoBufferTimestamps[video] = [videoStartTime, videoStartTime + (self.videoStartBuffer)]
                                    
                # Get experiments ---------------------------------------------
                
                experimentTimes = []; experimentNames = []
                trialKeep = []
                for i in range((int(df_subjectEmotionAnnotations['jstime'].max()) + 1 - (self.trialDuration)) // (self.trialShift)):
                    # For each Trial:
                    trialStart = i * (self.trialShift)
                    trialEnd = trialStart + (self.trialDuration)
                    df_currTrialEmotionAnnotations = df_subjectEmotionAnnotations[(df_subjectEmotionAnnotations['jstime'] > trialStart) &
                                                                                  (df_subjectEmotionAnnotations['jstime'] < trialEnd)]
                    trialVideos = df_currTrialEmotionAnnotations['video'].unique()
                    if trialEnd - videoBufferTimestamps[trialVideos[-1]][0] > (self.videoStartBuffer):
                        # do not consider trial if survey response time (trial end) happens to soon after a video begins
                        experimentTimes.append([trialStart, trialEnd])
                        videoNames = [self.videoIDtoName[video] for video in trialVideos]
                        experimentNames.append('__'.join(videoNames))
                        trialKeep.append(i)
                    elif self.debug:
                        print('skipping', trialStart, trialEnd, videoBufferTimestamps[trialVideos[-1]])
                
                if showPlots:
                    experimentTimesPlotting = np.array(experimentTimes)
                    plt.vlines(trialKeep,experimentTimesPlotting[:, 0], experimentTimesPlotting[:, 1], label="Trials")
                    labelFlag = True
                    for timeWindow in videoBufferTimestamps.values():
                        if labelFlag:
                            plt.axhspan(timeWindow[0], timeWindow[1], color='grey', label='Video Start Buffer')
                        else:
                            plt.axhspan(timeWindow[0], timeWindow[1], color='grey')
                        labelFlag = False
                    
                    plt.legend()
                    plt.xlabel('Trial #')
                    plt.ylabel('Time, seconds')
                    plt.title('Trial Time Visualization')
                    plt.show()
                    
                print(len(trialKeep), 'of', (int(df_subjectEmotionAnnotations['jstime'].max()) + 1 - (self.trialDuration)) // (self.trialShift), 'trials kept')
                    
                # Get Valence + Arousal (Surveys) -----------------------------
                
                labelFlag = True
                currentSurveyAnswersList = []; currentSurveyAnswerTimes = []
                for timeWindow in experimentTimes:
                    currentSurveyAnswerTimes.append(timeWindow[1])
                    # surveys aggregate last few seconds of time window
                    df_currTrialEmotionAnnotations = df_subjectEmotionAnnotations[(df_subjectEmotionAnnotations['jstime'] >= timeWindow[1] - (self.surveyAggregation / 2)) &
                                                                                  (df_subjectEmotionAnnotations['jstime'] <= timeWindow[1] + (self.surveyAggregation / 2))]
                    
                    trialValence, trialArousal = self.aggregateSurveysPerTrial(df_currTrialEmotionAnnotations)
                    currentSurveyAnswersList.append([trialValence, trialArousal])
                    if showPlots:
                        if labelFlag:
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['valence'] - trialValence, 
                                     marker='.', color='blue', label='Valence Error')
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['arousal'] - trialArousal,
                                     marker='.', color='orange', label='Arousal Error')
                        else:
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['valence'] - trialValence, color='blue', marker='.')
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['arousal'] - trialArousal, color='orange', marker='.')
                            
                        labelFlag = False
                
                if showPlots:
                    plt.axvline(x=self.surveyAggregation / 2, color='black', linestyle='-', label='Trial End Time')
                    plt.legend()
                    plt.xlabel('Time from survey aggregation start, seconds')
                    plt.ylabel('Delta from Aggregated Value')
                    plt.title(f"Valence and Arousal vs. Aggregated Value Error t={self.surveyAggregation} sec")
                    plt.show()
                
                # Get Signal Data ---------------------------------------------
                
                timePoints = list(df_subjectSynchronizedData['daqtime'].dropna())
                experimentSignalData = []; dataPerFreq = []
                for i, signalName in enumerate(self.deap_streamingOrder):
                    signalData = list(df_subjectSynchronizedData[signalName].dropna() * self.deap_streamingConversions[i])
                    assert len(timePoints) == len(signalData), f'Expected {len(timePoints)} number of datapoints. Got {len(signalData)}'
                    dataPerFreq.append(signalData)
                
                if showPlots:
                    for i, signalData in enumerate(dataPerFreq):
                        dataPlotting = signalData[20000:25000]
                        timePlotting = timePoints[20000:25000]
                        plt.plot(timePlotting, [(e-min(dataPlotting))/(max(dataPlotting)-min(dataPlotting)) for e in dataPlotting], 
                                 marker='.', linestyle='--', label=f'{self.deap_streamingOrder[i]} [{min(dataPlotting)}, {max(dataPlotting)}]')
                    plt.xlabel('Time, seconds')
                    plt.ylabel('Value, Normalized')
                    plt.legend()
                    plt.title('All Signals')
                    plt.show()
                experimentSignalData.append([timePoints, dataPerFreq])
                
                # Organize the data collected ---------------------------------
                
                # Convert times to seconds: 
                #DONE: @Jadelyn, would you be able to make everything seconds
                experimentTimes = np.array(experimentTimes)
                currentSurveyAnswerTimes = np.array(currentSurveyAnswerTimes)
                
                subjectOrder.append(subjectName)
                allExperimentalTimes.append(experimentTimes)
                allExperimentalNames.append(experimentNames)
                allSynchronizedData.append(experimentSignalData)
                allSurveyAnswerTimes.append(currentSurveyAnswerTimes)
                allSurveyAnswersList.append(currentSurveyAnswersList)
                
                print("\tFinished data extraction");
                
                if self.debug:
                    print(self.extractExperimentLabels(allExperimentalNames[0]))
                    
                    print(np.array(subjectOrder).shape)
                    print(np.array(allExperimentalTimes).shape)
                    print(np.array(allExperimentalNames).shape)
                    print(np.array(allSurveyAnswerTimes).shape)
                    print(np.array(allSurveyAnswersList).shape)
                
        return allSynchronizedData, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo

    def extractExperimentLabels(self, allExperimentalNames):
        activityNames = np.asarray(self.activityNames)
        
        activityLabels = []
        # For each trial during the day.
        for trialActivityName in allExperimentalNames:
            # Get the videos information.
            videoType = trialActivityName.split('__')[-1]
            
            # Store the videos as a hashed index.
            activityHash = next((i for i, name in enumerate(activityNames) if name in videoType))
            activityLabels.append(activityHash)

        return activityNames, activityLabels
    
    def getStreamingInfo(self):
        # Feature information
        streamingOrder = ['lowFreq', 'lowFreq', 'eda', 'lowFreq', 'temp', 'highfreq', 'highfreq', 'highfreq']
        biomarkerOrder = ['lowFreq', 'lowFreq', 'eda', 'lowFreq', 'temp', 'highfreq', 'highfreq', 'highfreq']
        filteringOrders = [[None, None], [None, 30], [None, 15], [None, 30], [None, 0.1], [None, None], [None, None], [None, None]]  # Sampling Freq: 1000 (Hz); Need 1/2 frequency at max.
        featureAverageWindows = [30, 30, 30, 30, 30, 30, 30, 30] # ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
        
        return streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders
    
    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = self.getStreamingInfo()
        featureNames, biomarkerFeatureNames, biomarkerOrder = self.compileFeatureNames.extractFeatureNames(biomarkerOrder)
            
        return streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames
        
        
if __name__ == "__main__":
    # Initialize metadata analysis class.
    deapAnalysisClass = deapInterface()
    
    analyzingData = True
    trainingData = False
    
    if analyzingData:
        # Extract the metadata
        allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, \
            allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo = deapAnalysisClass.getData(showPlots = True)
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = deapAnalysisClass.getStreamingInfo()
        # # Analyze and save the metadata features
        deapAnalysisClass.extractFeatures(allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                                                      streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders, interfaceType = 'deap', reanalyzeData = True, showPlots = True)
    
    if trainingData:
        # Prepare the data to go through the training interface.
        streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames = deapAnalysisClass.compileTrainingInfo()
        
        plotTrainingData = False
        # Collected the training data.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
                experimentOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = deapAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)
    
    