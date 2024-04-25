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

# Module to read .bdf files
import mne

# -------------------------------------------------------------------------- #
# ------------------------- Extract Data from Excel ------------------------ #
class deapInterface(_globalMetaAnalysis.globalMetaAnalysis):
    # Initialize location of data and parameters.
    def __init__(self):
        self.debug = False

        # Specify path to data.
        self.subjectFolders = (os.path.dirname(__file__)+ "/../../../Data/_metaDatasets/DEAP/")

        # List of relevant demographic information.
        self.demographicQuestions = ["Age", "Gender", "Education"]

        # Initialize DEAP survey information.
        self.surveyedEmotions = ["Valence", "Arousal", "Dominance", "Liking", "Familiarity"]

        # Compile survey questions.
        self.surveyQuestions = [em.lower() for em in self.surveyedEmotions]

        # Compile scoring ranges.
        self.surveyResponseRanges = [9]*(len(self.surveyedEmotions) - 1)
        self.surveyResponseRanges.extend([5]*1)

        # Compile sampling frequency and data points for physiological data.
        # NOTE: This is for the pre-processed data, which we probably won't use.
        self.samplingFreq = 128
        self.numPoints = 322560
        self.numChannels = 40 

        # Initialize the global meta protocol.
        super().__init__(
            self.subjectFolders, self.surveyQuestions
        )
    
    # ------------------------- HELPER FUNCTIONS FOR DATA POPULATION ------------------------ #

    # Populates subjectOrder.
    # Size: (numExp).
    # Example output: [1, 2, ...]
    def getSubjectOrder(self, surveyAnswer_df):
        return surveyAnswer_df.Participant_id.unique()


    # Populates getAllContextualInfo. 
    # Size: (numExp, 2, numDemographics).
    # Example: [[['Age', 'Gender', 'Education'], [24, 'Male, 'BA']], 
    #           [['Age', 'Gender', 'Education'], [33, 'Female', 'MA']],
    #           ...]
    def getAllContextualInfo(self, subjectOrder, contextual_df):
        allContextualInfo = []
        for pid in subjectOrder:
            pid_questions = self.demographicQuestions.copy()
            pid_responses = []
            pid_q = "S" + str(pid).zfill(2) # Format Participant ID as 'S01', 'S02', etc.
            for question in pid_questions:
                response = contextual_df.query("Participant_id==@pid_q")[question].iloc[0]
                if question == 'Age':
                    response = int(response)
                pid_responses.append(response)
            allContextualInfo.append([pid_questions, pid_responses])
        return allContextualInfo


    # Populate getAllExperimentTimes.
    # Size: (numExp, numTrials, 2).
    # Example: [[[0, 2], [4, 6], ... ] ... ]
    def getAllExperimentTimes(self, subjectOrder, surveyAnswer_df):
        # TODO: Uncertainty about what exactly experimental times mean; email unanswered
        allExperimentalTimes = []
        for pid in subjectOrder:
            numTrials = len(surveyAnswer_df[surveyAnswer_df["Participant_id"] == pid])
            startTimes = []
            for t in range(1, numTrials+1):
                startTime = surveyAnswer_df.query("Participant_id==@pid and Experiment_id==@t")["Start_time"].iloc[0]
                endTime = startTime + 6e+6 # For now, assume end time is 60s after start time
                startTimes.append([startTime, endTime])
            allExperimentalTimes.append(startTimes)
        return allExperimentalTimes
    

    # Populate allExperimentalNames.
    # Size: (numExp, numTrials).
    # Example: [Biking, Rest]
    def getAllExperimentalNames(self, subjectOrder, surveyAnswer_df, videoInfo_df):
        allExperimentalNames = []
        for pid in subjectOrder:
            numTrials = len(surveyAnswer_df[surveyAnswer_df["Participant_id"] == pid])
            trialNames = []
            for t in range(1, numTrials+1):
                trialVideoID = surveyAnswer_df.query("Participant_id==@pid and Experiment_id==@t")["Experiment_id"].iloc[0]
                videoName = videoInfo_df.query("Experiment_id==@trialVideoID")["Title"].iloc[0]
                trialName = "Participant " + str(pid) + ", Song: " + videoName
                trialNames.append(trialName)
            allExperimentalNames.append(trialNames)
        return allExperimentalNames
    

    # Populate allSurveyAnswerTimes.
    # Size: (numExp, numTrials).
    # Example: [2, 4, ... ]
    def getAllSurveyAnswerTimes(self, subjectOrder, surveyAnswer_df):
        # TODO: Uncertainty about what exactly experimental times mean; email unanswered
        # Since survey answer times are not listed, we assume survey answer time is experiment end time.
        allSurveyAnswerTimes = []
        for pid in subjectOrder:
            numTrials = len(surveyAnswer_df[surveyAnswer_df["Participant_id"] == pid])
            surveyAnswerTimes = []
            for t in range(1, numTrials+1):
                startTime = surveyAnswer_df.query("Participant_id==@pid and Experiment_id==@t")["Start_time"].iloc[0]
                endTime = startTime + 6e+6 # For now, assume end time is 60s after start time
                surveyAnswerTimes.append(endTime)
            allSurveyAnswerTimes.append(surveyAnswerTimes)
        return allSurveyAnswerTimes
    

    # Populate allSurveyAnswersList.
    # Size: (numExp, numTrials, numEmotions).
    # Example: [[0, 1, 2], [1, 0, 1] ... ]
    def getAllSurveyAnswersList(self, subjectOrder, surveyAnswer_df):
        allSurveyAnswersList = []
        for pid in subjectOrder:
            numTrials = len(surveyAnswer_df[surveyAnswer_df["Participant_id"] == pid])
            emotionRatings = []
            for t in range(1, numTrials+1):
                trialRatings = []
                for emotion in self.surveyedEmotions:
                    rating = surveyAnswer_df.query("Participant_id==@pid and Experiment_id==@t")[emotion].iloc[0]
                    trialRatings.append(rating)
                emotionRatings.append(trialRatings)
            allSurveyAnswersList.append(emotionRatings)
        return allSurveyAnswersList
    

    # NOTE: ERRORS DUE TO DISCONTINUITY!
    # Populate allCompiledData using preprocessed data.
    # Size: (numExp, numFreq, 2) where 2 = ((numTimePoints), (numSignals, numPoints))
    def getAllCompiledDataPreprocessed(self, subjectOrder, surveyAnswer_df):
        allCompiledData = []
        for pid in subjectOrder:
            timepoints_data = []
            # Add timepoints (measured at 128 Hz).
            signalDataTimepoints = self.universalMethods.getEvenlySampledArray(self.samplingFreq, self.numPoints)
            signalDataTimepoints = np.array(signalDataTimepoints)
            timepoints_data.append(signalDataTimepoints)

            # Initialize signals for this participant.
            allChannelsData = []

            # Open file with data for each participant.
            fileName = "s" + str(pid).zfill(2) + ".mat"
            signalDataFile = scipy.io.loadmat(self.subjectFolders + "data_preprocessed_matlab/" + fileName)
            allSignalData = signalDataFile["data"] # Signal data for this participant.

            # Get presentation order for participant.
            presentation_order = []
            numTrials = len(surveyAnswer_df[surveyAnswer_df["Participant_id"] == pid])
            for t in range(1, numTrials+1):
                trialVideoID = surveyAnswer_df.query("Participant_id==@pid and Trial==@t")["Video_id"].iloc[0]
                presentation_order.append(trialVideoID)

            for ch in range(self.numChannels):
                channelContinuousData = [] # All data, continuous, for this channel.
                # Traverse through data in presentation order for continuity.
                for tr in presentation_order:
                    tr = tr - 1 # 1-indexing to 0-indexing
                    channelTrialData = allSignalData[tr][ch]
                    for datapoint in channelTrialData:
                        channelContinuousData.append(datapoint)
                # BEGIN TEST
                if pid == 1 and ch == 38:
                    plt.plot(signalDataTimepoints, channelContinuousData, marker=".")
                    plt.axvline(x=63)
                    plt.axis([60, 66, -65000, 65000])
                    plt.show()
                # END TEST
                allChannelsData.append(channelContinuousData)

            allChannelsData = np.array(allChannelsData)
            timepoints_data.append(allChannelsData)
            print("PID: ", pid, " | timepoints shape: ", signalDataTimepoints.shape, " | data shape: ", 
                  allChannelsData.shape, " | timepoints_data length: ", len(timepoints_data))
            allCompiledData.append(timepoints_data)
        
        return allCompiledData


    # NOTE: Unfinished data compilation for raw data.
    # NOTE: Install mne in Terminal with: pip install mne
    def getAllCompiledDataRaw(self, subjectOrder, surveyAnswer_df):
        allCompiledData = []

        for pid in subjectOrder:
            fileName = self.subjectFolders + "data_original/" + "s" + str(pid).zfill(2) + ".bdf"
            raw_data = mne.io.read_raw_bdf(fileName, preload=True)

            # TODO: Length of raw data varies with each file.
            # print(len(raw_data))

        return allCompiledData


    # Extract data from file.
    def getData(self):
        # Initialize data holders.
        subjectOrder = []; allContextualInfo = []
        allExperimentalTimes = []; allExperimentalNames = []
        allSurveyAnswerTimes = []; allSurveyAnswersList = []
        allCompiledData = []

        # Gather relevant files.
        contextual_df = pd.read_excel(self.subjectFolders + "metadata_xls/participant_questionnaire.xls")
        videoInfo_df = pd.read_excel(self.subjectFolders + "metadata_xls/video_list.xls")
        surveyAnswer_df = pd.read_excel(self.subjectFolders + "metadata_xls/participant_ratings.xls")

        # Populate subjectOrder.
        surveyAnswer_df.sort_values(by=["Participant_id", "Experiment_id"], inplace=True)
        subjectOrder = self.getSubjectOrder(surveyAnswer_df)
        print("subjectOrder size: ", np.array(subjectOrder).shape)

        # Populate allContextualInfo.
        allContextualInfo = self.getAllContextualInfo(subjectOrder, contextual_df)
        print("allContextualInfo size: ", np.array(allContextualInfo).shape)

        # Populate allExperimentalTimes.
        allExperimentalTimes = self.getAllExperimentTimes(subjectOrder, surveyAnswer_df)
        print("allExperimentalTimes size: ", np.array(allExperimentalTimes).shape)

        # Populate allExperimentalNames with experiment names.
        allExperimentalNames = self.getAllExperimentalNames(subjectOrder, surveyAnswer_df, videoInfo_df)
        print("allExperimentalNames size: ", np.array(allExperimentalNames).shape)

        # Populate allSurveyAnswerTimes.
        allSurveyAnswerTimes = self.getAllSurveyAnswerTimes(subjectOrder, surveyAnswer_df)
        print("allSurveyAnswerTimes size: ", np.array(allSurveyAnswerTimes).shape)

        # Populate allSurveyAnswersList
        allSurveyAnswersList = self.getAllSurveyAnswersList(subjectOrder, surveyAnswer_df)
        print("allSurveyAnswersList size: ", np.array(allSurveyAnswersList).shape)
        
        # Populate allCompiledData
        allCompiledData = self.getAllCompiledDataRaw(subjectOrder, surveyAnswer_df)

        if self.debug:
            print("\n subjectOrder: ", subjectOrder, "\n")
            print("\n allContextualInfo: ", allContextualInfo, "\n")
            print("\n allExperimentalTimes: ", allExperimentalTimes, "\n")
            print("\n allExperimentalNames: ", allExperimentalNames, "\n")
            print("\n allSurveyAnswerTimes: ", allSurveyAnswerTimes)
            print("\n allSurveyAnswersList: ", allSurveyAnswersList, "\n")
            print("\n allCompiledData: ", allCompiledData, "\n")

        # TODO: Go into files and remove bad values (need allCompiledData to work to do this).
            

if __name__ == "__main__":
    # Initialize metadata analysis class.
    deapAnalysisClass = deapInterface()
    deapAnalysisClass.getData()
