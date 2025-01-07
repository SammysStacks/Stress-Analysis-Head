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

    def closeDeviceStream(self):
        if self.deviceType == "empatica":
            self.deviceReader.mainDevice.closeServer = True
        if hasattr(self, "serverThread"): self.serverThread.join()
        self.mainDevice.close()

    def streamWearableData(self, adcResolution, stopTimeStreaming, filePath):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Beginning to stream in data!")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Get user information
        self.setUserName(filePath)

        # Prepare the arduino to stream in data
        self.stopTimeStreaming = self.setupDeviceStream(stopTimeStreaming, usingTimestamps=False)
        streamingDataFingers = [0 for _ in range(self.numUniqueSignals)]
        highestSamplingFreqTimes = self.analysisList[0].timepoints

        try:
            # Loop Through and Read the Arduino Data in Real-Time
            while len(highestSamplingFreqTimes) == 0 or (highestSamplingFreqTimes[-1] - highestSamplingFreqTimes[0]) < self.stopTimeStreaming:
                # Collect and compile the most recently available data points.
                if self.deviceType == "serial": self.recordData(self.voltageRange[1], adcResolution)

                # Analyze the new data in batches.
                streamingDataFingers = self.analyzeBatchData(streamingDataFingers)

            #self.analyzeBatchData(streamingDataFingers)

        except Exception as error:
            self.closeDeviceStream()
            print("Streaming error:", error)

        finally:
            # Set the final experimental time to the end of the experiment
            if len(self.experimentTimes) != 0 and self.experimentTimes[-1][1] is None:
                self.experimentTimes[-1][1] = highestSamplingFreqTimes[-1]

            # Close the Arduino's at the End
            print("\nFinished Streaming in Data; Closing Arduino\n")
            self.closeDeviceStream()

    def streamExcelData(self, compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, filePath):
        print("\tAnalyzing the Excel Data")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Add Experimental information
        self.subjectInformationQuestions = subjectInformationQuestions
        self.subjectInformationAnswers = subjectInformationAnswers
        self.surveyQuestions = surveyQuestions
        # Experiment information parameters
        self.experimentInfoPointerStart = 0
        self.experimentInfoPointerEnd = 0
        self.featureInfoPointer = 0
        # Get user information
        self.setUserName(filePath)

        # Extract the Time and Voltage Data
        timepoints, biomarkerData = compiledRawData
        biomarkerData = np.asarray(biomarkerData)
        # Compile streaming parameters.
        self.numPointsPerBatch = min(self.numPointsPerBatch, len(timepoints))
        streamingDataFingers = [0 for _ in range(self.numUniqueSignals)]
        excelDataFinger = 0

        # Loop Through and Read the Excel Data in Pseudo-Real-Time
        while excelDataFinger != len(timepoints):
            self.organizeData(timepoints=timepoints[excelDataFinger:excelDataFinger + self.moveDataFinger], datapoints=biomarkerData[:, excelDataFinger:excelDataFinger + self.moveDataFinger])
            excelDataFinger = min(len(timepoints), excelDataFinger + self.moveDataFinger)

            # When enough data has been collected, analyze the new data in batches.
            streamingDataFingers = self.analyzeBatchData(streamingDataFingers)

            # Organize experimental information.
            self.organizeExperimentalInformation(timepoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, excelDataFinger)

        if experimentTimes[-1][-1] < timepoints[-1]:
            # Assert that experimental information was read in correctly.
            assert np.array_equal(surveyAnswerTimes, self.surveyAnswerTimes), print(surveyAnswerTimes, self.surveyAnswerTimes)
            assert np.array_equal(surveyAnswersList, self.surveyAnswersList), print(surveyAnswersList, self.surveyAnswersList)
            assert np.array_equal(experimentTimes, self.experimentTimes), f"{experimentTimes} != {self.experimentTimes}"
            assert np.all(np.asarray(experimentNames) == np.asarray(self.experimentNames)), experimentNames

        # Finished Analyzing the Data
        print("\n\tFinished Analyzing Excel Data")

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
            print("\tWaiting for arduino to initialize!")
            time.sleep(1)
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
