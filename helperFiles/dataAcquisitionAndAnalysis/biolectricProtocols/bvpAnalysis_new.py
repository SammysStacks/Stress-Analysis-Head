import sklearn
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis

from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.globalProtocol import globalProtocol

class bvpProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # Feature collection parameters
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.featureTimeWindow = None  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

        # Filter parameters.
        self.cutOffFreq = [0.05, 20]  # Band pass filter frequencies.

        # Reset analysis variables
        super().__init__("bvp", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.
        self.featureTimeWindow = self.featureTimeWindow_highFreq  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

    def checkParams(self):
        pass

    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, 100)
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq * self.featureTimeWindow * 4 / 5)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxBufferSeconds))  # cutOffFreq = 0.1, use 70 seconds; cutOffFreq = 0.01, use 400 seconds; cutOffFreq = 0.05, use 100 seconds

    # ------------------------- Data Analysis Begins ------------------------ #

    def analyzeData(self, dataFinger):

        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):

            # ---------------------- Filter the Data ----------------------- #

            # Find the starting/ending points of the data to analyze
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.asarray(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.asarray(self.timePoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints=False)

            # ---------------------- Feature Extraction --------------------- #

            if self.collectFeatures:
                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timePoints):
                    featureTime = self.timePoints[self.lastAnalyzedDataInd[channelIndex]]

                    # Find the start window pointer.
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    # Compile the good data in the feature interval.
                    intervalTimes, intervalData = self.compileBatchData(filteredTime, filteredData, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer[channelIndex], channelIndex)

                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatch < len(intervalTimes):
                        # Calculate and save the features in this window.
                        finalFeatures = self.extractFeatures(intervalTimes, intervalData)

                        # Keep track of the new features.
                        newRawFeatures.append(finalFeatures)  # Dimension: [numTimePoints x numChannelFeatures]
                        newFeatureTimes.append(featureTime)  # Dimension: [numTimePoints]

                    # Keep track of which data has been analyzed
                    self.lastAnalyzedDataInd[channelIndex] = lastPulseIndex

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

            # -------------------------------------------------------------- #

    def filterData(self, timePoints, data, removePoints=False):
        # Filter the Data: Low pass Filter and Savgol Filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low', fastFilt = True)
        filteredTime = timePoints.copy()

        return filteredTime, filteredData, np.ones(len(filteredTime))

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window
        while self.timePoints[timePointer] < currentTime - timeWindow:
            timePointer += 1

        return timePointer

    def compileBatchData(self, filteredTime, filteredData, goodIndicesMask, startFilterPointer, startFeatureTimePointer, channelIndex):
        assert len(goodIndicesMask) >= len(filteredData) == len(filteredTime), print(len(goodIndicesMask), len(filteredData), len(filteredTime))

        # Accounts for the missing points (count the number of viable points within each pointer).
        startReferenceFinger = (goodIndicesMask[0:startFeatureTimePointer - startFilterPointer]).sum(axis=0, dtype=int)
        endReferenceFinger = startReferenceFinger + (goodIndicesMask[startFeatureTimePointer - startFilterPointer:self.lastAnalyzedDataInd[channelIndex] + 1 - startFilterPointer]).sum(axis=0, dtype=int)
        # Compile the information in the interval.
        intervalTimes = filteredTime[startReferenceFinger:endReferenceFinger]
        intervalData = filteredData[startReferenceFinger:endReferenceFinger]

        return intervalTimes, intervalData

    @staticmethod
    def independentComponentAnalysis(data):
        ica = sklearn.decomposition.FastICA(whiten='unit-variance')
        data = ica.fit_transform(data.reshape(-1, 1)).transpose()[0]

        return data

    # --------------------- Feature Extraction Methods --------------------- #

    def extractFeatures(self, timePoints, data):
        # ----------------------- Data Preprocessing ----------------------- #
        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)

        # ----------------------- Feature Extraction ----------------------- #
        # Extract the features from the data
        featureList = []
        # Compute the first and second derivatives
        first_derivative = np.gradient(standardized_data)
        second_derivative = np.gradient(first_derivative)

        systolic_peaks, _ = find_peaks(standardized_data, distance=64)  # return the indices of the peaks
        end_of_cycle, _ = find_peaks(-standardized_data, distance=64)  # return the indices of the local minima

        # Identify dicrotic notches and diastolic peaks within each pulse
        dicrotic_notches = []
        diastolic_peaks = []
        pulse_widths = []
        pulse_amplitudes = []

        for i in range(len(systolic_peaks) - 1):
            # Segment between two systolic peaks
            start_idx = systolic_peaks[i]
            next_idx = systolic_peaks[i + 1]
            segment = standardized_data[start_idx:next_idx]
            segment_time = timePoints[start_idx:next_idx]
            segment_first_derivative = first_derivative[start_idx:next_idx]
            segment_second_derivative = second_derivative[start_idx:next_idx]

            # Identify the dicrotic notch using the second derivative zero-crossing
            zero_crossings = np.where(np.diff(np.sign(segment_second_derivative)))[0]
            if zero_crossings.size > 0:
                # Assume the first zero-crossing after the systolic peak is the notch
                notch_idx = zero_crossings[0] + start_idx
                dicrotic_notches.append(notch_idx)
            else:
                dicrotic_notches.append(None)

            # Identify diastolic peak as a local maximum after the dicrotic notch
            if dicrotic_notches[-1] is not None:
                diastolic_segment = standardized_data[dicrotic_notches[-1]:next_idx]
                diastolic_peaks_in_segment, _ = find_peaks(diastolic_segment)
                if diastolic_peaks_in_segment.size > 0:
                    diastolic_peak_idx = diastolic_peaks_in_segment[0] + dicrotic_notches[-1]
                    diastolic_peaks.append(diastolic_peak_idx)
                else:
                    diastolic_peaks.append(None)
            else:
                diastolic_peaks.append(None)

            # Calculate pulse width and amplitudes
            width = timePoints[end_of_cycle[i + 1]] - timePoints[end_of_cycle[i]]
            pulse_widths.append(width)
            if diastolic_peaks[i] is not None:
                # Pulse magnitude is the difference between systolic and diastolic peak values
                magnitude = standardized_data[systolic_peaks[i]] - standardized_data[end_of_cycle[i]]
                pulse_amplitudes.append(magnitude)
            else:
                pulse_amplitudes.append(None)

        #  Heart Rate (HR)
        if len(systolic_peaks) > 1:
            hr_intervals = np.diff(timePoints[systolic_peaks])
            hr = 60 / np.mean(hr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(hr_intervals) ** 2))
        else:
            hr = None
            rmssd = None

        # Skewness and Kurtosis of the data
        data_skewness = skew(standardized_data)
        data_kurtosis = kurtosis(standardized_data)

        featureList.extend([systolic_peaks, diastolic_peaks, dicrotic_notches, pulse_widths, pulse_amplitudes])
        featureList.extend([hr, data_skewness, data_kurtosis, rmssd])

        return featureList
\
