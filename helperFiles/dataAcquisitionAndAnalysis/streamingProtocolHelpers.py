import collections
import threading

import numpy as np

# Import Bioelectric Analysis Files
from .biolectricProtocols.eogAnalysis import eogProtocol
from .biolectricProtocols.eegAnalysis import eegProtocol
from .biolectricProtocols.ecgAnalysis import ecgProtocol
from .biolectricProtocols.edaAnalysis import edaProtocol
from .biolectricProtocols.emgAnalysis import emgProtocol
from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.bvpAnalysis import bvpProtocol
from .biolectricProtocols.accelerationAnalysis import accelerationProtocol
from .biolectricProtocols.temperatureAnalysis import tempProtocol
from .biolectricProtocols.generalAnalysis_lowFreq import generalProtocol_lowFreq
from .biolectricProtocols.generalAnalysis_highFreq import generalProtocol_highFreq

# Import Modules to Read in Data
from .humanMachineInterface.serialInterface import serialInterface  # Functions to Read in Data from Arduino
from .humanMachineInterface.featureOrganization import featureOrganization

# Import plotting protocols
from .dataVisualization.biolectricPlottingProtocols import plottingProtocols
from .humanMachineInterface.serverInterface import serverInterface


# Parameters for the streamingProtocolHelpers class:
#     Biomarker information:
#         streamingOrder: a list of incoming biomarkers in the order they appear; Dim: numStreamedSignals
#         analysisOrder: a unique list of biomarkers in streamingOrder; Dim: numUniqueSignals
#
#     Channel information:
#         numChannelDist: The integer number of channels used in each analysis; Dim: numChannels_perAnalysis
#         channelDist: A dictionary of arrays, specifying each biomarker's indices in streamingOrder; Dim: numUniqueSignals, numChannels_perAnalysis


class streamingProtocolHelpers(featureOrganization):

    def __init__(self, deviceType, mainSerialNum, therapySerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, extractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData):
        # General streaming parameters.
        self.streamingOrder = np.char.lower(streamingOrder)  # The names of each recorded signal in order. Ex: ['eog', 'eog', 'eeg', 'eda']
        self.numStreamedSignals = len(streamingOrder)  # The total number of signals being recorded.
        self.numPointsPerBatch = numPointsPerBatch  # The number of points to analyze in each batch.
        self.plotStreamedData = plotStreamedData  # Boolean: whether to graph the incoming signals + analysis.
        self.moveDataFinger = moveDataFinger  # The minimum number of NEW points to analyze in each batch.
        self.voltageRange = voltageRange  # The voltage range of the incoming signals.
        self.deviceType = deviceType  # The type of device being used.
        self.mainDevice = None  # The main device being used.

        # Specify the analysis order: a unique list of biomarkers in streamingOrder.
        self.analysisOrder = list(collections.OrderedDict.fromkeys(self.streamingOrder))  # The set of unique biomarkers, maintaining the order they will be analyzed. Ex: ['eog', 'eeg', 'eda']
        self.numUniqueSignals = len(self.analysisOrder)  # The number of unique biomarkers being recorded.

        # Variables that rely on the incoming data's order.
        self.numChannelDist = np.zeros(self.numUniqueSignals)  # Track the number of channels used by each sensor.
        self.channelDist = {}  # Track the order (its index) each sensor comes in. Dim: numUniqueSignals. numChannelsPerBiomarker

        # Populate the variables, accounting for the order.
        for analysisInd in range(len(self.analysisOrder)):
            biomarkerType = self.analysisOrder[analysisInd]

            # Find the locations where each biomarker appears.
            streamingChannelIndices = self.streamingOrder == biomarkerType

            # Organize the streaming channels by their respective biomarker.
            self.numChannelDist[analysisInd] = np.sum(streamingChannelIndices)
            self.channelDist[biomarkerType] = np.where(streamingChannelIndices)[0]
        assert np.sum(self.numChannelDist) == self.numStreamedSignals, f"The number of channels per biomarker ({self.numChannelDist}) does not align with the streaming order ({self.streamingOrder})"

        # Initialize global plotting class.
        print(self.plotStreamedData)
        self.plottingClass = plottingProtocols(self.numStreamedSignals, self.channelDist, self.analysisOrder) if self.plotStreamedData else None
        self.analysisProtocols = {
            'highfreq': generalProtocol_highFreq(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['highfreq'], self.plottingClass, self) if 'highfreq' in self.analysisOrder else None,
            'lowfreq': generalProtocol_lowFreq(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['lowfreq'], self.plottingClass, self) if 'lowfreq' in self.analysisOrder else None,
            'temp': tempProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['temp'], self.plottingClass, self) if 'temp' in self.analysisOrder else None,
            'eog': eogProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eog'], self.plottingClass, self, voltageRange) if 'eog' in self.analysisOrder else None,
            'eeg': eegProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eeg'], self.plottingClass, self) if 'eeg' in self.analysisOrder else None,
            'ecg': ecgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['ecg'], self.plottingClass, self) if 'ecg' in self.analysisOrder else None,
            'eda': edaProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eda'], self.plottingClass, self) if 'eda' in self.analysisOrder else None,
            'emg': emgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['emg'], self.plottingClass, self) if 'emg' in self.analysisOrder else None,
            'bvp': bvpProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['bvp'], self.plottingClass, self) if 'bvp' in self.analysisOrder else None,
            'acc': accelerationProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['acc'], self.plottingClass, self) if 'acc' in self.analysisOrder else None,
        }

        self.analysisList = []
        # Generate a list of all analyses, keeping the streaming order.
        for biomarkerType in self.analysisOrder:
            self.analysisList.append(self.analysisProtocols[biomarkerType])

        # Store the deviceReader Instance
        if self.deviceType == 'serial':
            self.deviceReader = serialInterface(mainSerialNum=mainSerialNum, therapySerialNum=therapySerialNum)
            self.mainDevice = self.deviceReader.mainDevice
        elif self.deviceType == 'empatica':
            self.deviceReader = serverInterface(streamingOrder, self.analysisProtocols, deviceType=deviceType)
            self.serverThread = threading.Thread(target=self.deviceReader.startServer, daemon=True)
            self.mainDevice = self.deviceReader.mainDevice
            self.serverThread.start()
        else: raise ValueError(f"Device Type {deviceType} is not recognized.")

        # Holder parameters.
        self.subjectInformationQuestions = None  # A list of subject background questions
        self.subjectInformationAnswers = None  # A list of subject background answers, where each element represents an answer to subjectInformationQuestions.
        self.surveyAnswersList = None  # A list of lists of survey answers, where each element represents an answer to surveyQuestions.
        self.surveyAnswerTimes = None  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = None  # A list of survey questions, where each element in surveyAnswersList corresponds to this question order.
        self.experimentTimes = None  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = None  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).

        # Finish setting up the class.
        super().__init__(modelClasses, actionControl, self.analysisProtocols, extractFeaturesFrom, featureAverageWindows)
        self.resetStreamingInformation()

    def resetStreamingInformation(self):
        self.resetFeatureInformation()
        # Reset the analysis information
        for analysis in self.analysisList:
            analysis.resetAnalysisVariables()

        # Subject Information
        self.subjectInformationQuestions = []  # A list of subject background questions, such as race, age, and gender.
        self.subjectInformationAnswers = []  # A list of subject background answers, where each element represents an answer to subjectInformationQuestions.

        # Survey Information
        self.surveyAnswersList = []  # A list of lists of survey answers, where each element represents a list of answers to surveyQuestions.
        self.surveyAnswerTimes = []  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = []  # A list of survey questions, where each element in surveyAnswersList corresponds to this question order.

        # Experimental information
        self.experimentTimes = []  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = []  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).

    def analyzeBatchData(self, streamingDataFinger):
        # Analyze the current data
        for analysis in self.analysisList:
            analysis.analyzeData(streamingDataFinger)

        # Organize the new features
        self.organizeRawFeatures()
        self.alignFeatures()
        # self.predictLabels()

        # Plot the Data
        if self.plotStreamedData: self.plottingClass.displayData()

        # Move the streamingDataFinger pointer to analyze the next batch of data
        return streamingDataFinger + self.moveDataFinger

    def recordData(self, maxVolt=3.3, adcResolution=4096):
        assert self.deviceType == "serial", f"Recording data is only supported for serial devices, not {self.deviceType}."

        rawReadsList = []
        # Read in at least one point
        while int(self.mainDevice.in_waiting) > 0 or len(rawReadsList) == 0:
            rawReadsList.append(self.deviceReader.readline(ser=self.mainDevice))

        # Parse the Data
        timepoints, datapoints = self.deviceReader.parseCompressedRead(rawReadsList, self.numStreamedSignals, maxVolt, adcResolution)
        self.organizeData(timepoints, datapoints)  # Organize the data for further processing

    def organizeData(self, timepoints, datapoints):
        if len(timepoints) == 0: print("\tNO NEW timepoints ADDED")

        if not isinstance(datapoints, list): datapoints = list(datapoints)
        if not isinstance(timepoints, list): timepoints = list(timepoints)

        # Update the data (if present) for each sensor
        for analysisInd in range(len(self.analysisOrder)):
            analysis = self.analysisList[analysisInd]

            # Skip if no data is present for this sensor
            if analysis.numChannels == 0: continue

            # Update the timepoints.
            analysis.timepoints.extend(timepoints)

            # For each channel, update the voltage data.
            for channelIndex in range(analysis.numChannels):
                # Compile the datapoints for each of the sensor's channels.
                streamingDataIndex = analysis.streamingChannelInds[channelIndex]
                newData = datapoints[streamingDataIndex]

                # Add the Data to the Correct Channel
                analysis.channelData[channelIndex].extend(newData)
