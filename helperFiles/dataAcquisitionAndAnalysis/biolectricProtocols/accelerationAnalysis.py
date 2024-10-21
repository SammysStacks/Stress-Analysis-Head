import sklearn
import scipy
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis

from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.globalProtocol import globalProtocol

class accelerationProtocol(globalProtocol):
    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        """Note: for acceleration data analysis, the 3 axis channel data will be streamed in as a single channel by calculating the magnitudes"""
        # Feature collection parameters
        """sliding windows definition"""
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.featureTimeWindow = None  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

        # Filter parameters.
        self.cutOffFreq = [0.05, 10]  # Band pass filter frequencies.

        # Reset analysis variables
        super().__init__("acc", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
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

            timepoints = np.asarray(self.timepoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timepoints, dataBuffer, removePoints=False)

            # ---------------------- Feature Extraction --------------------- #

            if self.collectFeatures:
                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timepoints):
                    featureTime = self.timepoints[self.lastAnalyzedDataInd[channelIndex]]

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
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatureTimes[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

            # -------------------------------------------------------------- #

    def filterData(self, timepoints, data, removePoints=False):
        # Filter the Data: Low pass Filter and Savgol Filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq, self.samplingFreq, order=3, filterType='bandpass', fastFilt=True)
        filteredTime = timepoints.copy()

        return filteredTime, filteredData, np.ones(len(filteredTime))

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window
        while self.timepoints[timePointer] < currentTime - timeWindow:
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

    @staticmethod
    def zero_crossing_rate(data):
        return ((data[:-1] * data[1:]) < 0).sum()

    @staticmethod
    def entropy(data):
        # shannon entropy
        value, counts = np.unique(data, return_counts=True)
        return -np.sum(counts / len(data) * np.log2(counts / len(data)))

    def band_energy(self, data, band=(2, 3)):
        f, Pxx = welch(data, fs=self.samplingFreq, nperseg=1024)
        band_power = scipy.integrate.simps(Pxx[(f >= band[0]) & (f <= band[1])])
        return band_power

    def spectral_flux(self, data):
        freqs, Pxx = welch(data, fs=self.samplingFreq, nperseg=1024)
        flux = np.sqrt(np.mean(np.diff(Pxx) ** 2))
        return flux

    def extractFeatures(self, timepoints, data):
        # ----------------------- Data Preprocessing ----------------------- #
        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)

        # ----------------------- Feature Extraction ----------------------- #
        # Extract the features from the data
        featureList = []

        # mean, std
        mean_acc = np.mean(standardized_data)
        std_acc = np.std(standardized_data)

        # kurtosis, skewness
        kurtosis_acc = kurtosis(standardized_data)
        skewness_acc = skew(standardized_data)

        # zero crossing rate; entropy; band energy; spectral flux
        zcr = self.zero_crossing_rate(standardized_data)
        entropy = self.entropy(standardized_data)
        band_energy = self.band_energy(standardized_data)
        spectral_flux = self.spectral_flux(standardized_data)

        featureList.extend([mean_acc, std_acc, kurtosis_acc, skewness_acc, zcr, entropy, band_energy, spectral_flux])

        return featureList



