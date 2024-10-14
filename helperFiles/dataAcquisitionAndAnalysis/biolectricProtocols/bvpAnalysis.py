import math

import matplotlib.pyplot as plt
import numpy as np
# Baseline Subtraction
from scipy.signal import savgol_filter

from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.globalProtocol import globalProtocol
from BaselineRemoval import BaselineRemoval

class bvpProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # Feature collection parameters
        self.diastolicPressure0 = None
        self.systolicPressure0 = None
        self.featureListAverage = None
        self.featureListExact = None
        self.maxPointsPerPulse = None
        self.minPointsPerPulse = None
        self.peakStandard = 0  # The Max First Deriviative of the Previous Pulse's Systolic Peak
        self.peakStandardInd = 0  # The Index of the Max Derivative in the Previous Pulse's Systolic Peak

        # Pointer initialization
        self.timeOffset = None
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.prevEndInd = None
        self.endIndexPointer = None

        # general parameters
        self.featureTimeWindow = None  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

        self.plottingIndicator = False  # Plot the Data
        # Filter parameters.
        self.cutOffFreq = [0.05, 5]  # Band pass filter frequencies.

        # Parameters that Define a Pulse
        self.maxBPM = 180  # The Maximum Beats Per Minute for a Human; max = 480
        self.minBPM = 60  # The Minimum Beats Per Minute for a Human; min = 27
        # Pulse Separation Parameters
        self.bufferTime = 60 / self.minBPM  # The Initial Wait Time Before we Start Labeling Peaks

        self.previousSystolicAmp = None

        # Reset analysis variables
        super().__init__("bvp", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):

        # document the lastDataIndex over the batch
        self.timeOffset = 0

        # document the absolute timer
        self.prevEndInd = 0
        self.endIndexPointer = 0

        # General parameters
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.
        self.featureTimeWindow = self.featureTimeWindow_highFreq  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

        # Feature Lists
        self.featureListExact = []  # List of Lists of Features; Each Index Represents a Pulse; Each Pulse's List Represents its Features
        self.featureListAverage = []  # List of Lists of Features Averaged in Time by self.numSecondsAverage; Each Index Represents a Pulse; Each Pulse's List Represents its Features

        # Peak Seperation Parameters
        self.peakStandard = 0  # The Max First Deriviative of the Previous Pulse's Systolic Peak
        self.peakStandardInd = 0  # The Index of the Max Derivative in the Previous Pulse's Systolic Peak



    def checkParams(self):
        pass

    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, 100)
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq * self.featureTimeWindow * 4 / 5)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxBufferSeconds))

    def filterData(self, timepoints, data, removePoints=False):
        # Filter the Data: Band-pass Filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(
            data, self.cutOffFreq, self.samplingFreq, order=3, filterType='band', fastFilt=True
        )
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
        from sklearn.decomposition import FastICA
        ica = FastICA(whiten='unit-variance')
        data = ica.fit_transform(data.reshape(-1, 1)).transpose()[0]

        return data

    def setPressureCalibration(self, systolicPressure0, diastolicPressure0):
        self.systolicPressure0 = systolicPressure0
        self.diastolicPressure0 = diastolicPressure0

    def separatePulses(self, time, firstDer):
        self.peakStandardInd = 0

        # Take First Derivative of the Smooth
        # ened Data
        separatedPeaks = []
        for pointInd in range(len(firstDer)): # keep track of the current point index
            # Retrieve the derivative at pointInd
            firstDerVal = firstDer[pointInd]

            # If the derivative is greater than the threshold, then we have a potential peak following
            if firstDerVal > self.peakStandard*0.5:
                # Use the First few peaks as a standard
                if (self.timeOffset != 0 or 1.5 < time[pointInd]) and self.minPointsPerPulse < pointInd:
                    # if the point is sufficiently far away from the last peak, then we can consider it a peak
                    if self.peakStandardInd + self.minPointsPerPulse < pointInd: # checks whether current Index is far enough from the last separation peak index
                        separatedPeaks.append(pointInd)

                    # else: find the max of the peak
                    elif firstDer[separatedPeaks[-1]] < firstDer[pointInd]:
                        separatedPeaks[-1] = pointInd # peak refinement, make sure we identify the systolic peak is the actual peak

                    # else do not update the pointInd
                    else:
                        continue

                    self.peakStandardInd = pointInd
                    self.peakStandard = firstDerVal

                else:
                    self.peakStandard = max(self.peakStandard, firstDerVal)
        return separatedPeaks

    @staticmethod
    def retrieveTimePoints(timepoints, startPointer, endPointer):
        return timepoints[startPointer:endPointer]

    def analyzeData(self, dataFinger):
        for channelIndex in range(self.numChannels):
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.asarray(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timepoints = np.asarray(self.timepoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)
                self.setSamplingFrequencyParams()
                self.minPointsPerPulse = math.floor(self.samplingFreq * 60 / self.maxBPM)
                self.maxPointsPerPulse = math.ceil(self.samplingFreq * 60 / self.minBPM)

            filteredTime, filteredData, goodIndicesMask = self.filterData(timepoints, dataBuffer, removePoints=False)
            standardizeData = self.universalMethods.standardizeData(filteredData)
            standardizeData = savgol_filter(standardizeData, 11, 3)

            if self.collectFeatures:
                newFeatureTimes, newRawFeatures = [], []

                # Loop through the data, processing it in chunks, while making sure we analyze the whole batch.
                iteration = 0
                while self.lastAnalyzedDataInd[channelIndex] < len(timepoints):
                    featureTime = self.timepoints[self.lastAnalyzedDataInd[channelIndex]]
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(
                        self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    print('_____________________________defining intervalTimes____________________________________________')
                    intervalTimes, intervalData = self.compileBatchData(
                        filteredTime, standardizeData, goodIndicesMask, startFilterPointer,
                        self.startFeatureTimePointer[channelIndex], channelIndex)

                    # If the interval data or times are empty, skip further analysis.
                    if len(intervalData) == 0 or len(intervalTimes) == 0:
                        print("Interval data or times is empty. Skipping analysis.")
                        self.lastAnalyzedDataInd[channelIndex] += self.numPointsPerBatch
                        continue

                    first_derivative = np.gradient(intervalData, intervalTimes)
                    second_derivative = np.gradient(first_derivative, intervalTimes)
                    third_derivative = np.gradient(second_derivative, intervalTimes)

                    separatedPeaks = self.separatePulses(intervalTimes, first_derivative)

                    # If no peaks are found, adjust the peakStandard and attempt to detect again.
                    while len(separatedPeaks) == 0:
                        self.peakStandard /= 2
                        separatedPeaks = self.separatePulses(intervalTimes, first_derivative)

                    # Start pulse separation and update index for each pulse processed.
                    pulseStartInd = self.universalMethods.findNearbyMinimum(
                        intervalData, separatedPeaks[0], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse
                    )

                    for pulseNum in range(1, len(separatedPeaks)):
                        pulseEndInd = self.universalMethods.findNearbyMinimum(
                            intervalData, separatedPeaks[pulseNum], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse
                        )
                        print('pulseStartInd', pulseStartInd)
                        print('pulseEndInd', pulseEndInd)
                        # Validate pulse size and skip the pulse if it's too big or too small.
                        if pulseEndInd - pulseStartInd > self.maxPointsPerPulse:
                            print('Pulse too big; skipping.')
                            # print out the time points skipped
                            # print('Timepoints skipped:', intervalTimes[pulseStartInd:pulseEndInd])
                            pulseStartInd = pulseEndInd
                            continue
                        elif pulseEndInd - pulseStartInd < self.minPointsPerPulse:
                            print('Pulse too small; skipping.')
                            pulseStartInd = pulseEndInd
                            continue

                        intervalPulseTime = intervalTimes[pulseStartInd:pulseEndInd]
                        intervalPulseData = intervalData[pulseStartInd:pulseEndInd]

                        if iteration == 0:
                            absoluteTime = self.timepoints[pulseStartInd:pulseEndInd]
                        else:
                            shiftedStartInd = pulseStartInd + self.prevEndInd
                            shiftedEndInd = pulseEndInd + self.prevEndInd
                            absoluteTime = self.timepoints[shiftedStartInd:shiftedEndInd]

                        # print('intervalTimes', intervalTimes)
                        # print('intervalPulseTime', intervalPulseTime)


                        if len(intervalPulseTime) >= self.minPointsPerPulse:
                            toBeNormalizedPulse = intervalPulseData.copy()
                            normalizedPulseData = self.normalizePulseBaseline(toBeNormalizedPulse, 1)
                            finalFeatures = self.extractPulsePeaks(intervalPulseTime, normalizedPulseData, first_derivative, second_derivative, third_derivative)

                            if finalFeatures is not None:
                                newRawFeatures.append(finalFeatures)
                                newFeatureTimes.append(featureTime)

                                # Plot each pulse's features if plotting is enabled.
                                if self.plottingIndicator:
                                    self.plotBvpFeatures(absoluteTime, intervalPulseData, finalFeatures)

                            # Update the last analyzed index for this pulse.
                            self.timeOffset = pulseEndInd - pulseStartInd
                            self.endIndexPointer = pulseEndInd


                    self.lastAnalyzedDataInd[channelIndex] += self.timeOffset
                    print('self.lastAnalyzedDataInd', self.lastAnalyzedDataInd)
                    iteration += 1
                    self.prevEndInd += self.endIndexPointer

                if self.readData is None:
                    print('readData is None, doing initial Testings')
                else:
                    self.readData.compileContinuousFeatures(
                        newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex],
                        self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex],
                        self.featureAverageWindow
                    )

    def extractFeatures(self, normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allDicroticPeaks):

        featureList = [
            allSystolicPeaks, allDicroticPeaks
        ]
        return featureList


    def extractPulsePeaks(self, pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv):

        # ----------------------- Detect Systolic Peak ---------------------- #
        # Find Systolic Peak
        systolicPeakInd = self.universalMethods.findNearbyMaximum(normalizedPulse, 0, binarySearchWindow=4, maxPointsSearch=len(pulseTime))
        # Find UpStroke Peaks
        systolicUpstrokeVelInd = self.universalMethods.findNearbyMaximum(pulseVelocity, 0, binarySearchWindow=1, maxPointsSearch=systolicPeakInd)
        systolicUpstrokeAccelMaxInd = self.universalMethods.findNearbyMaximum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow=-1, maxPointsSearch=systolicPeakInd)
        systolicUpstrokeAccelMinInd = self.universalMethods.findNearbyMinimum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow=1, maxPointsSearch=systolicPeakInd)
        # ------------------------------------------------------------------- #

        # ----------------------  Detect Dicrotic Peak ---------------------- #
        dicroticNotchInd = self.universalMethods.findNearbyMinimum(normalizedPulse, systolicPeakInd, binarySearchWindow=1, maxPointsSearch=int(len(pulseTime) / 2))
        dicroticPeakInd = self.universalMethods.findNearbyMaximum(normalizedPulse, dicroticNotchInd, binarySearchWindow=1, maxPointsSearch=int(len(pulseTime) / 2))

        # Other Extremas Nearby
        dicroticInflectionInd = self.universalMethods.findNearbyMaximum(pulseVelocity, dicroticNotchInd, binarySearchWindow=2, maxPointsSearch=int(len(pulseTime) / 2))
        dicroticFallVelMinInd = self.universalMethods.findNearbyMinimum(pulseVelocity, dicroticInflectionInd, binarySearchWindow=2, maxPointsSearch=int(len(pulseTime) / 2))


        # ----------------------- Sequence Validation ----------------------- #
        # Check if the sequence is valid: systolic upstroke -> systolic peak -> dicrotic notch -> dicrotic peak
        if not (systolicUpstrokeVelInd < systolicPeakInd < dicroticNotchInd < dicroticPeakInd):
            print("Invalid peak sequence detected. Skipping pulse.")
            return None  # Skip this pulse if the sequence is incorrect

        # ----------------------- Value Validation -------------------------- #
        # check if the systolic peak is higher than diastolic peak
        if not (normalizedPulse[systolicPeakInd] > normalizedPulse[dicroticPeakInd]):
            print("Invalid systolic and diastolic pulse value. Skipping pulse.")
            return None

        # ----------------------- Feature Extraction ------------------------ #
        allSystolicPeaks = [systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd]
        allDicroticPeaks = [dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd]


        # Extract the Pulse Features
        featureList = self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allDicroticPeaks)


        return featureList

        # ------------------------------------------------------------------- #

    def calculateHeartRate(self, timepoints, systolic_peaks):
        if len(systolic_peaks) > 1:
            systolic_peak_times = timepoints[systolic_peaks]
            rr_intervals = np.diff(systolic_peak_times)
            hr = 60 / np.mean(rr_intervals)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        else:
            hr = None
            rmssd = None
        return hr, rmssd

    def plotBvpFeatures(self, timepoints, bvp_data, features):
        #print('timepoints', timepoints)
        systolic_peaks = features[0]
        dicrotic_info = features[1]

        systolic_peak = systolic_peaks[3]  # Systolic Peak
        dicrotic_notch = dicrotic_info[0]  # Dicrotic Notch
        dicrotic_peak = dicrotic_info[2]  # Dicrotic Peak

        # Start plotting
        plt.figure(figsize=(10, 6))
        plt.plot(timepoints, bvp_data, label="BVP Signal", color="blue")

        # Plot and label the systolic peaks
        # print(f"\n[DEBUG] Plotting systolic peak at index: {systolic_peak}")
        plt.plot(timepoints[systolic_peak], bvp_data[systolic_peak], 'ro', label="Systolic Peak")
        plt.text(timepoints[systolic_peak], bvp_data[systolic_peak], 'Systolic', color="red", fontsize=10, ha="center")

        # Plot and label the dicrotic notches
        # print(f"\n[DEBUG] Plotting dicrotic notch at index: {dicrotic_notch}")
        plt.plot(timepoints[dicrotic_notch], bvp_data[dicrotic_notch], 'yo', label="Dicrotic Notch")
        plt.text(timepoints[dicrotic_notch], bvp_data[dicrotic_notch], 'Notch', color="yellow", fontsize=10, ha="center")

        # Plot and label the dicrotic peaks
        # print(f"\n[DEBUG] Plotting dicrotic peak at index: {dicrotic_peak}")
        plt.plot(timepoints[dicrotic_peak], bvp_data[dicrotic_peak], 'bo', label="Dicrotic Peak")
        plt.text(timepoints[dicrotic_peak], bvp_data[dicrotic_peak], 'Dicrotic', color="blue", fontsize=10, ha="center")

        # Final plot adjustments
        plt.xlabel("Time (s)")
        plt.ylabel("BVP Signal (a.u.)")
        plt.title("BVP Signal with Detected Peaks")
        plt.legend()
        plt.grid(True)
        plt.show()



    # ---------------------------- Data preprocessing ----------------------------
    @staticmethod
    def normalizeMinMax(data):
        return data / np.max(np.abs(data))

    def normalizePulseBaseline(self, pulseData, polynomialDegree):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            pulseData:  yData-Axis Data for a Single Pulse (Start-End)
            polynomialDegree: Polynomials Used in Baseline Subtraction
        Output Parameters:
            pulseData: yData-Axis Data for a Baseline-Normalized Pulse (Start, End = 0)
        Use Case: Shift the Pulse to the xData-Axis (Removing non-Horizontal Base)
        Assumption in Function: pulseData is Positive
        ----------------------------------------------------------------------
        Further API Information Can be Found in the Following Link:
        https://pypi.org/project/BaselineRemoval/
        ----------------------------------------------------------------------
        """
        # Perform Baseline Removal Twice to Ensure Baseline is Gone
        for _ in range(2):
            # Baseline Removal Procedure
            baseObj = BaselineRemoval(pulseData)  # Create Baseline Object
            pulseData = baseObj.ModPoly(polynomialDegree)  # Perform Modified multi-polynomial Fit Removal

        # Return the Data With Removed Baseline
        return pulseData



