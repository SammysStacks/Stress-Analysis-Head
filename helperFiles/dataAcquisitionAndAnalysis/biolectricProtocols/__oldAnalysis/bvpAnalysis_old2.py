import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis

from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.globalProtocol import globalProtocol


class bvpProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # Feature collection parameters
        self.maxPointsPerPulse = None
        self.minPointsPerPulse = None
        self.peakStandard = 0  # The Max First Deriviative of the Previous Pulse's Systolic Peak
        self.peakStandardInd = 0  # The Index of the Max Derivative in the Previous Pulse's Systolic Peak
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.featureTimeWindow = None  # The duration of time that each feature considers
        self.minPointsPerBatch = None  # The minimum number of points that must be present in a batch to extract features.

        self.plottingIndicator = False  # Plot the Data
        # Filter parameters.
        self.cutOffFreq = [0.05, 5]  # Band pass filter frequencies.

        # Parameters that Define a Pulse
        self.maxBPM = 180  # The Maximum Beats Per Minute for a Human; max = 480
        self.minBPM = 40  # The Minimum Beats Per Minute for a Human; min = 27
        # Pulse Separation Parameters
        self.bufferTime = 60 / self.minBPM  # The Initial Wait Time Before we Start Labeling Peaks

        self.previousSystolicAmp = None

        # Reset analysis variables
        super().__init__("bvp", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters
        self.timeOffset = 0
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

            if self.collectFeatures:
                newFeatureTimes, newRawFeatures = [], []
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timepoints):
                    featureTime = self.timepoints[self.lastAnalyzedDataInd[channelIndex]]
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(
                        self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)

                    intervalTimes, intervalData = self.compileBatchData(
                        filteredTime, filteredData, goodIndicesMask, startFilterPointer,
                        self.startFeatureTimePointer[channelIndex], channelIndex)

                    if self.minPointsPerBatch < len(intervalTimes):
                        finalFeatures = self.extractFeatures(intervalTimes, intervalData)
                        newRawFeatures.append(finalFeatures)
                        newFeatureTimes.append(featureTime)

                        print(f'finalFeatures: {finalFeatures}')

                        # plot for each batch
                        if self.plottingIndicator:
                            self.plotBvpFeatures(intervalTimes, intervalData, finalFeatures)

                    self.lastAnalyzedDataInd[channelIndex] += len(find_peaks(intervalData)[0])
                if self.readData is None:
                    print('readData is None, doing initial Testings')
                else:
                    self.readData.compileContinuousFeatures(
                        newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex],
                        self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex],
                        self.featureAverageWindow)

    @staticmethod
    def normalizeMinMax(data):
        return data / np.max(np.abs(data))

    def extractFeatures(self, timepoints, data):
        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        # might not be necessary, but doesnt change the shape of the data much
        standardized_data = savgol_filter(standardized_data, 11, 3)

        # Compute first and second derivatives
        first_derivative = np.gradient(standardized_data, timepoints)
        second_derivative = np.gradient(first_derivative, timepoints)

        plt.plot(timepoints, self.normalizeMinMax(standardized_data), 'k', linewidth=2, alpha=1, label='Standardized BVP Data')
        plt.plot(timepoints, self.normalizeMinMax(first_derivative), 'tab:red', linewidth=1.5, alpha=1, label='First Derivative')
        plt.plot(timepoints, self.normalizeMinMax(second_derivative), 'tab:blue', linewidth=1.5, alpha=0.75, label='Second Derivative')
        plt.plot(timepoints, np.zeros_like(timepoints), 'tab:red', linewidth=1.5, alpha=1)

        # Add labels and legend
        plt.xlabel('Time (s)')
        plt.ylabel('BVP Data (Normalized)')
        plt.legend(loc='best')

        # Show the plot
        plt.show()

        exit()
        # Prepare feature collection
        systolic_peaks, bad_systolic_peak_idx, dicrotic_notches, diastolic_peaks = [], [], [], []
        pulse_widths, pulse_amplitudes = [], []

        pointer = 0
        iteration = 0

        separatedPeaks = systolic_peaks.extend(self.seperatePulses(timepoints, first_derivative))

        while pointer < len(standardized_data):
            # Identify the pulse start using the first derivative (negative to positive transition)
            pulse_start = self.find_pulse_start(pointer, first_derivative)
            if pulse_start is None:
                break

            # Identify the systolic peak as a local maximum
            systolic_peak, potential_bad_idx = self.find_systolic_peak(pulse_start, first_derivative, standardized_data)

            print('systolic_peak: ', systolic_peak)
            if systolic_peak is None:
                break
            if potential_bad_idx is not None:
                bad_systolic_peak_idx.append(potential_bad_idx)
                # append the bad index any way, we will remove the bad index later
                systolic_peaks.append(systolic_peak)

                # skip to the pulse end
                pulse_end = self.find_pulse_end(potential_bad_idx, None, None, first_derivative)
                pointer = pulse_end
            else:
                systolic_peaks.append(systolic_peak)
                # Identify dicrotic notch using second derivative zero-crossing or local minimum
                dicrotic_notch_ind = self.findDicroticNotch(first_derivative, second_derivative, systolic_peak)
                dicrotic_notches.append(dicrotic_notch_ind)

                # Identify diastolic peak or fallback to systolic peak
                if dicrotic_notch_ind is not None:
                    diastolic_peak = self.findDiastolicPeak(standardized_data, dicrotic_notch_ind, len(standardized_data))
                else:
                    diastolic_peak = self.findDiastolicPeak(standardized_data, systolic_peak, len(standardized_data))
                diastolic_peaks.append(diastolic_peak)

                # Identify pulse end based on first derivative (positive to negative transition)
                pulse_end = self.find_pulse_end(systolic_peak, dicrotic_notch_ind, diastolic_peak, first_derivative)
                # pulse end should always be identified unless its the last pulse (complete pulse or not)
                if pulse_end is None:
                    pulse_end = len(standardized_data)
                pulse_end = min(pulse_end, len(standardized_data) - 1)
                pointer = pulse_end

            # Calculate amplitude and width
            amplitude = standardized_data[systolic_peak] - np.min(standardized_data[pulse_start:pulse_end])
            width = timepoints[pulse_end] - timepoints[pulse_start]

            pulse_amplitudes.append(amplitude)
            pulse_widths.append(width)
            iteration += 1

        hr, rmssd = self.calculateHeartRate(timepoints, systolic_peaks)

        data_skewness = skew(standardized_data)
        data_kurtosis = kurtosis(standardized_data)


        featureList = [
            systolic_peaks, diastolic_peaks, dicrotic_notches, pulse_widths, pulse_amplitudes, hr,
            data_skewness, data_kurtosis, rmssd
        ]

        return featureList

    # to find the pulse start based on first derivative
    def find_pulse_start(self, pointer, first_derivative):
        # here I define we always find the pulse start from the 1st first derivative sign change from negative to positive
        # which then indicates the start of the pulse
        for i in range(pointer, len(first_derivative)):
            if first_derivative[i] > 0 and first_derivative[i - 1] <= 0:
                return i
        return None

    # to find the systolic peak
    def find_systolic_peak(self, pulse_start, first_derivative, standardized_data):
        local_max = np.max(standardized_data[pulse_start:pulse_start + int(self.samplingFreq * 0.7)])  # Assume pulse lasts up to 0.7s
        dynamic_threshold = 0.5 * local_max
        bad_idx = None
        for i in range(pulse_start, len(first_derivative)):
            if first_derivative[i] < 0 and first_derivative[i - 1] > 0:
                if standardized_data[i] > dynamic_threshold:
                    return i, bad_idx
                else:
                    print(f"Systolic peak found at index {i}, but below dynamic threshold.")
                    bad_idx = i
                    return bad_idx, bad_idx
        return None, None

    def seperatePulses(self, time, firstDer):
        self.peakStandardInd = 0
        # Take First Derivative of Smoothened Data
        systolicPeaks = []
        for pointInd in range(len(firstDer)):
            # Calcuate the Derivative at pointInd
            firstDerVal = firstDer[pointInd]

            # If the Derivative Stands Out, Its the Systolic Peak
            if firstDerVal > self.peakStandard * 0.5:

                # Use the First Few Peaks as a Standard
                if (self.timeOffset != 0 or 1.5 < time[pointInd]) and self.minPointsPerPulse < pointInd:
                    # If the Point is Sufficiently Far Away, its a New R-Peak
                    if self.peakStandardInd + self.minPointsPerPulse < pointInd:
                        systolicPeaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif firstDer[systolicPeaks[-1]] < firstDer[pointInd]:
                        systolicPeaks[-1] = pointInd
                    # Else, Dont Update Pointer
                    else:
                        continue
                    self.peakStandardInd = pointInd
                    self.peakStandard = firstDerVal
                else:
                    self.peakStandard = max(self.peakStandard, firstDerVal)

        return systolicPeaks

    # to find pulse end based on first derivative
    def find_pulse_end(self, systolic_peak, dicrotic_notch_ind, diastolic_peak, first_derivative):
        if diastolic_peak is not None:
            start_idx = diastolic_peak
        elif dicrotic_notch_ind is not None:
            # only if the diastolic peak is not identified, then we start with dicrotic notch
            start_idx = dicrotic_notch_ind
        else:
            # if neither dicrotic notch nor diastolic peak is identified, we fallback to systolic peak
            start_idx = systolic_peak

        for i in range(start_idx, len(first_derivative)):
            if first_derivative[i] > 0 and first_derivative[i - 1] < 0:
                return i
        return None

    def findDicroticNotch(self, first_derivative, second_derivative, systolic_peak):
        """
        Identify the dicrotic notch using second derivative zero-crossing or local minimum.
        """
        zero_crossings = np.where(np.diff(np.sign(second_derivative[systolic_peak:])))
        if zero_crossings[0].size > 0:
            inflectionPoint = zero_crossings[0][0] + systolic_peak # we only want to identify the first zero-crossing
            return inflectionPoint
            # for dicroticNotchInd in range(inflectionPoint, len(first_derivative)):
            #     if first_derivative[dicroticNotchInd] > 0 and first_derivative[dicroticNotchInd - 1] <= 0:
            #         return dicroticNotchInd
        return None

    def findDiastolicPeak(self, standardized_data, start_idx, end_idx):
        """
        Find the diastolic peak after the dicrotic notch or systolic peak (only if dicrotic notch was not identified)
        """
        segment = standardized_data[start_idx:end_idx]
        diastolic_peaks, _ = find_peaks(segment)
        if diastolic_peaks.size > 0:
            return diastolic_peaks[0] + start_idx
        return None

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
        systolic_peaks = features[0]
        diastolic_peaks = features[1]
        dicrotic_notches = features[2]

        # Debugging: Print out the features before filtering and plotting
        print(f"\n[DEBUG] Features before filtering:")
        print(f"Systolic Peaks (raw): {systolic_peaks}")
        print(f"Diastolic Peaks (raw): {diastolic_peaks}")
        print(f"Dicrotic Notches (raw): {dicrotic_notches}")

        # Filter out None values from systolic peaks, diastolic peaks, and dicrotic notches
        systolic_peaks = [idx for idx in systolic_peaks if idx is not None]
        diastolic_peaks = [idx for idx in diastolic_peaks if idx is not None]
        dicrotic_notches = [idx for idx in dicrotic_notches if idx is not None]

        # Debugging: Print out the features after filtering
        print(f"\n[DEBUG] Features after filtering:")
        print(f"Systolic Peaks (filtered): {systolic_peaks}")
        print(f"Diastolic Peaks (filtered): {diastolic_peaks}")
        print(f"Dicrotic Notches (filtered): {dicrotic_notches}")

        # Start plotting
        plt.figure(figsize=(10, 6))
        plt.plot(timepoints, bvp_data, label="BVP Signal", color="blue")

        # Plot and label the systolic peaks
        if systolic_peaks:
            print(f"\n[DEBUG] Plotting systolic peaks at indices: {systolic_peaks}")
            for idx in systolic_peaks:
                print(f"Systolic Peak at index {idx}, Time: {timepoints[idx]}")
            plt.plot(timepoints[systolic_peaks], bvp_data[systolic_peaks], 'ro', label="Systolic Peaks")
            # for idx in systolic_peaks:
            #     plt.text(timepoints[idx], bvp_data[idx], color="red", fontsize=12, ha="center")

        # Plot and label the diastolic peaks
        if diastolic_peaks:
            print(f"\n[DEBUG] Plotting diastolic peaks at indices: {diastolic_peaks}")
            for idx in diastolic_peaks:
                print(f"Diastolic Peak at index {idx}, Time: {timepoints[idx]}")
            plt.plot(timepoints[diastolic_peaks], bvp_data[diastolic_peaks], 'go', label="Diastolic Peaks")
            # for idx in diastolic_peaks:
            #     plt.text(timepoints[idx], bvp_data[idx], color="green", fontsize=12, ha="center")

        # Plot and label the dicrotic notches
        if dicrotic_notches:
            print(f"\n[DEBUG] Plotting dicrotic notches at indices: {dicrotic_notches}")
            for idx in dicrotic_notches:
                print(f"Dicrotic Notch at index {idx}, Time: {timepoints[idx]}")
            plt.plot(timepoints[dicrotic_notches], bvp_data[dicrotic_notches], 'yo', label="Dicrotic Notches")
            # for idx in dicrotic_notches:
            #     plt.text(timepoints[idx], bvp_data[idx], color="yellow", fontsize=12, ha="center")

        # Final plot adjustments
        plt.xlabel("Time (s)")
        plt.ylabel("BVP Signal (a.u.)")
        plt.title("BVP Signal with Detected Peaks")
        plt.legend()
        plt.grid(True)
        plt.show()

        # exit()


