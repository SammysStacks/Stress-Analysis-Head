import math
import time
from datetime import datetime

import numpy as np

# Import helper files.
from .streamingProtocolHelpers import streamingProtocolHelpers


class streamingProtocols(streamingProtocolHelpers):

    def __init__(self, deviceType, mainSerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, extractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData):
        # Create Pointer to Common Functions
        super().__init__(deviceType=deviceType, mainSerialNum=mainSerialNum, therapySerialNum=None, modelClasses=modelClasses, actionControl=actionControl, numPointsPerBatch=numPointsPerBatch, moveDataFinger=moveDataFinger,
                         streamingOrder=streamingOrder, extractFeaturesFrom=extractFeaturesFrom, featureAverageWindows=featureAverageWindows, voltageRange=voltageRange, plotStreamedData=plotStreamedData)
        # Holder parameters.
        self.experimentInfoPointerStart = None
        self.experimentInfoPointerEnd = None
        self.featureInfoPointer = None
        self.stopTimeStreaming = None

    def resetGlobalVariables(self):
        self.resetStreamingInformation()
        # Reset the Experiment Information Pointers
        self.experimentInfoPointerStart = None
        self.experimentInfoPointerEnd = None
        self.featureInfoPointer = None
        self.stopTimeStreaming = None

    def setupDeviceStream(self, stopTimeStreaming, usingTimestamps=False):
        # self.deviceReader.resetArduino(self.mainDevice, 10)
        # Read and throw out the first few reads
        if self.deviceType == "serial":
            rawReadsList = []
            while int(self.mainDevice.in_waiting) > 0 or len(rawReadsList) < 2000:
                rawReadsList.append(self.deviceReader.readline(ser=self.mainDevice))
        elif self.deviceType == "empatica":
            self.deviceReader.mainDevice.closeServer = False

        if usingTimestamps:
            # Calculate the Stop Time
            timeBuffer = 0
            if type(stopTimeStreaming) in [float, int]:
                # Save Time Buffer
                timeBuffer = stopTimeStreaming
                # Get the Current Time as a TimeStamp
                currentTime = datetime.now().time()
                stopTimeStreaming = str(currentTime).replace(".", ":")
            # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
            stopTimeStreaming = self.convertToTime(stopTimeStreaming) + timeBuffer

        return stopTimeStreaming

    def streamWearableData(self, adcResolution, stopTimeStreaming, filePath):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Beginning to stream in data!")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Get user information
        self.setUserName(filePath)

        # Prepare the arduino to stream in data
        self.stopTimeStreaming = self.setupDeviceStream(stopTimeStreaming, usingTimestamps=False)
        timepoints = self.analysisList[0].timepoints
        streamingDataFinger = 0

        try:
            # Loop Through and Read the Arduino Data in Real-Time
            while len(timepoints) == 0 or (timepoints[-1] - timepoints[0]) < self.stopTimeStreaming:
                # Collect and compile the most recently available data points.
                if self.deviceType == "serial": self.recordData(self.voltageRange[1], adcResolution)

                # When enough data has been collected, analyze the new data in batches.
                while len(timepoints) - streamingDataFinger >= self.numPointsPerBatch:
                    streamingDataFinger = self.analyzeBatchData(streamingDataFinger)

            # At the end, analyze all remaining data
            self.analyzeBatchData(streamingDataFinger)

        except Exception as error:
            self.mainDevice.close()
            print(error)

        finally:
            # Set the final experimental time to the end of the experiment
            if len(self.experimentTimes) != 0 and self.experimentTimes[-1][1] is None:
                self.experimentTimes[-1][1] = timepoints[-1]

            # Close the Arduino's at the End
            print("\nFinished Streaming in Data; Closing Arduino\n")
            self.mainDevice.close()

    def streamExcelData(self, deviceType, compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes,
                        surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, filePath):
        print("\tAnalyzing the Excel Data")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Add Experimental information
        self.surveyQuestions = surveyQuestions
        self.subjectInformationAnswers = subjectInformationAnswers
        self.subjectInformationQuestions = subjectInformationQuestions
        # Experiment information parameters
        self.experimentInfoPointerStart = 0
        self.experimentInfoPointerEnd = 0
        self.featureInfoPointer = 0
        # Get user information
        self.setUserName(filePath)

        # Extract the Time and Voltage Data
        if deviceType == "serial":
            """timepoints = []
                Voltages = [[], [], [], []] with same length as timepoints """
            timepoints, Voltages = compiledRawData
            Voltages = np.asarray(Voltages)
            # Compile streaming parameters.
            numPointsPerBatch = min(self.numPointsPerBatch, len(timepoints))
            streamingDataFinger = 0
            excelDataFinger = 0

            # Loop Through and Read the Excel Data in Pseudo-Real-Time
            while excelDataFinger != len(timepoints):
                self.organizeData(deviceType, timepoints=timepoints[excelDataFinger:excelDataFinger + self.moveDataFinger], datapoints=Voltages[:, excelDataFinger:excelDataFinger + self.moveDataFinger])
                excelDataFinger = min(len(timepoints), excelDataFinger + self.moveDataFinger)

                # When enough data has been collected, analyze the new data in batches.
                while numPointsPerBatch <= excelDataFinger - streamingDataFinger:
                    streamingDataFinger = self.analyzeBatchData(streamingDataFinger)
                # Organize experimental information.
                self.organizeExperimentalInformation(timepoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, excelDataFinger)

            # Assert that experimental information was read in correctly.

            assert np.array_equal(experimentTimes, self.experimentTimes), f"{experimentTimes} \n {self.experimentTimes}"
            assert np.all(np.asarray(experimentNames) == np.asarray(self.experimentNames)), experimentNames
            # Assert that experimental information was read in correctly.
            assert np.array_equal(surveyAnswerTimes, self.surveyAnswerTimes), print(surveyAnswerTimes, self.surveyAnswerTimes)
            assert np.array_equal(surveyAnswersList, self.surveyAnswersList), print(surveyAnswersList, self.surveyAnswersList)

            # Finished Analyzing the Data
            print("\n\tFinished Analyzing Excel Data")

        elif deviceType == "empatica":
            """Note: timePoints [[t1], [t2], [t3], [t4]]
                    Voltages: [[d1], [d2], [d3], [d4]]"""
            biomarkerIndex = 0 # 0 for acc, 1 for bvp, 2, for eda, 3 for temp
            # cannot process it this way
            for pairInd in range (0, len(compiledRawData[0])):
                timepoints = compiledRawData[0][pairInd]
                Voltages = compiledRawData[1][pairInd]
                Voltages = np.asarray(Voltages)
                numPointsPerBatch = min(self.numPointsPerBatch, len(timepoints))
                streamingDataFinger = 0
                excelDataFinger = 0

                # Loop Through and Read the Excel Data in Pseudo-Real-Time
                while excelDataFinger != len(timepoints):
                    self.organizeData(deviceType, timepoints=timepoints[excelDataFinger:excelDataFinger + self.moveDataFinger], datapoints=Voltages[excelDataFinger:excelDataFinger + self.moveDataFinger])
                    excelDataFinger = min(len(timepoints), excelDataFinger + self.moveDataFinger)

                    # When enough data has been collected, analyze the new data in batches.
                    while numPointsPerBatch <= excelDataFinger - streamingDataFinger:
                        streamingDataFinger = self.analyzeBatchData_e4(biomarkerIndex, streamingDataFinger)
                    # Organize experimental information.
                    self.organizeExperimentalInformation(timepoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, excelDataFinger)
                # Assert that experimental information was read in correctly.
                assert np.array_equal(experimentTimes, self.experimentTimes), f"{experimentTimes} \n {self.experimentTimes}"
                assert np.all(np.asarray(experimentNames) == np.asarray(self.experimentNames)), experimentNames
                # Assert that experimental information was read in correctly.
                assert np.array_equal(surveyAnswerTimes, self.surveyAnswerTimes), print(surveyAnswerTimes, self.surveyAnswerTimes)
                assert np.array_equal(surveyAnswersList, self.surveyAnswersList), print(surveyAnswersList, self.surveyAnswersList)

                # Finished Analyzing the Data
                print(f"\n\tFinished Analyzing Excel Data for pairs {pairInd + 1}")
                biomarkerIndex += 1


    def organizeExperimentalInformation(self, timepoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, excelDataFinger):
        # Add the experiment information when the timepoint is reached.
        while self.experimentInfoPointerStart != len(experimentTimes) and experimentTimes[self.experimentInfoPointerStart][0] <= timepoints[min(len(timepoints) - 1, excelDataFinger - 1)]:
            self.experimentTimes.append([experimentTimes[self.experimentInfoPointerStart][0], None])
            self.experimentNames.append(experimentNames[self.experimentInfoPointerStart])
            self.experimentInfoPointerStart += 1
        while self.experimentInfoPointerEnd != len(experimentTimes) and experimentTimes[self.experimentInfoPointerEnd][1] <= timepoints[min(len(timepoints) - 1, excelDataFinger - 1)]:
            self.experimentTimes[self.experimentInfoPointerEnd][1] = experimentTimes[self.experimentInfoPointerEnd][1]
            self.experimentInfoPointerEnd += 1
        # Add the feature information when the timepoint is reached.
        while self.featureInfoPointer != len(surveyAnswerTimes) and surveyAnswerTimes[self.featureInfoPointer] <= timepoints[min(len(timepoints) - 1, excelDataFinger - 1)]:
            self.surveyAnswerTimes.append(surveyAnswerTimes[self.featureInfoPointer])
            self.surveyAnswersList.append(surveyAnswersList[self.featureInfoPointer])
            self.featureInfoPointer += 1

    def getCurrentTime(self):
        # Return the last streaming timePoint
        while len(self.analysisList[0].timepoints) == 0:
            print("\tWaiting for arduino to initialize!"); time.sleep(1)
        return self.analysisList[0].timepoints[-1]

    @staticmethod
    def convertToTime(timeStamp):
        if isinstance(timeStamp, str):
            timeStamp = timeStamp.split(":")
        timeStamp.reverse()

        currentTime = 0
        orderOfInput = [1E-6, 1, 60, 60 * 60, 60 * 60 * 24]
        for i, timeStampVal in enumerate(timeStamp):
            currentTime += orderOfInput[i] * int(timeStampVal)
        return currentTime

    @staticmethod
    def convertToTimeStamp(timeSeconds):
        hours = timeSeconds // 3600
        remainingTime = timeSeconds % 3600
        minutes = remainingTime // 60
        remainingTime %= 60
        seconds = math.floor(remainingTime)
        microSeconds = remainingTime - seconds
        microSeconds = np.round(microSeconds, 6)
        return hours, minutes, seconds, microSeconds
