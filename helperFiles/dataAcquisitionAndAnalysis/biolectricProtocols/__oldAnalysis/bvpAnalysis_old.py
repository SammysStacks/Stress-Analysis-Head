import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis
import antropy as ant
from sklearn.preprocessing import MinMaxScaler

class bvpProtocol:
    def __init__(self, timePointer, signal, samplingFrequency, windowSizes, overlap=0.25):
        self.timePointer = timePointer
        self.signal = signal
        self.samplingFrequency = samplingFrequency
        self.windowSizes = windowSizes
        self.overlap = overlap
        # HardCoded Feature Names
        self.featureNames = ["Heart Rate", "RMSSD", "Pulse Amplitude", "Pulse Width", "Systolic Peaks", "Diastolic Peaks",
                         "Entropy", "DFA", "Mean", "Variance", "Skewness", "Kurtosis"]



    @staticmethod
    def normalizeWindowedFeatures(features_all_windows_list):
        # takes in a list of all windowed features
        scaler = MinMaxScaler()
        features_all_windows_normalized = []
        for features in features_all_windows_list:
            features_normalized = scaler.fit_transform(features)
            features_all_windows_normalized.append(features_normalized)
        return features_all_windows_normalized

    @staticmethod
    def rmssd(signal):
        """ RMSSD is a measure of HRV, which reflects the balance between sympathetic and parasympathetic nervous systems."""
        return np.sqrt(np.mean(np.square(np.diff(signal))))

    @staticmethod
    def pulse_amplitude(systolic_peaks, diastolic_peaks):
        """ Ensure systolic and diastolic peaks have the same length by using the shorter array. valid approach?
            systolic peaks are associated with pressure during contraction, diasolic peaks are associated with pressure during relaxation."""
        min_length = min(len(systolic_peaks), len(diastolic_peaks))
        return systolic_peaks[:min_length] - diastolic_peaks[:min_length]

    @staticmethod
    def pulse_width(timePointer, systolic_peaks, diastolic_peaks):
        """ Ensure systolic and diastolic peaks have the same length by using the shorter array. valid approach?"""
        min_length = min(len(systolic_peaks), len(diastolic_peaks))
        return timePointer[systolic_peaks[:min_length]] - timePointer[diastolic_peaks[:min_length]]

    def extract_features(self, signal, timePointer, systolic_peaks, diastolic_peaks):
        heart_rate = len(systolic_peaks) / (timePointer[-1] - timePointer[0]) * 60 # heart rate in bpm
        rmssd_value = self.rmssd(np.diff(systolic_peaks))
        pulse_amp = np.mean(self.pulse_amplitude(signal[systolic_peaks], signal[diastolic_peaks]))
        pulse_wid = np.mean(self.pulse_width(timePointer, systolic_peaks, diastolic_peaks))
        systolic_value = np.mean(signal[systolic_peaks])
        diastolic_value = np.mean(signal[diastolic_peaks])

        entropy_value = ant.perm_entropy(signal)
        dfa_value = ant.detrended_fluctuation(signal)

        mean_bvp = np.mean(signal)
        variance_bvp = np.var(signal)
        skewness_bvp = skew(signal)
        kurtosis_bvp = kurtosis(signal)

        return [heart_rate, rmssd_value, pulse_amp, pulse_wid, systolic_value, diastolic_value,
                entropy_value, dfa_value, mean_bvp, variance_bvp, skewness_bvp, kurtosis_bvp]

    def sliding_windowed_features(self, signal, timePointer, single_window_size, overlap=0.25):
        window_samples = int(single_window_size * self.samplingFrequency)
        step_size = int(window_samples * (1 - overlap))

        feature_list = []
        for dataPointer in range(0, len(signal) - window_samples + 1, step_size):
            window_signal = signal[dataPointer:dataPointer + window_samples]
            window_time = timePointer[dataPointer:dataPointer + window_samples]
            systolic_peaks, _ = find_peaks(window_signal, distance=30)
            diastolic_peaks, _ = find_peaks(-window_signal, distance=30)
            features = self.extract_features(window_signal, window_time, systolic_peaks, diastolic_peaks)
            feature_list.append(features)

        return feature_list

    def get_time_vector(self, signal_length, singleWindowSize):
        window_samples = int(singleWindowSize * self.samplingFrequency)  # Convert window size to samples
        step_size = int(window_samples * (1 - self.overlap))  # Calculate step size in samples
        num_windows = (signal_length - window_samples) // step_size + 1  # Calculate number of windows
        time_vector = np.arange(0, num_windows) * (step_size / self.samplingFrequency)  # Convert step size to time in seconds
        return time_vector

    def analyzeData(self):
        # compile the sliding windowed features for all window sizes
        features_all_windows = []
        for window_size in self.windowSizes:
            features = self.sliding_windowed_features(self.signal, self.timePointer, window_size, self.overlap)
            features_all_windows.append(features)
        normalizedFeatures = self.normalizeWindowedFeatures(features_all_windows)

        # define the time vector
        time_vector = []
        for window_size in self.windowSizes:
            time_vector.append(self.get_time_vector(len(self.signal), window_size))

        # assert each windowed feature list has the same length as each time windows list within the time_vector list
        for features, time in zip(normalizedFeatures, time_vector):
            print(f'length of features: {len(features)}, length of time: {len(time)}')
            assert len(features) == len(time), "Mismatch in length between windowed features and time windows"

        return time_vector, normalizedFeatures

    def plotting_and_saving_features(self, time_vector, normalizedFeatures, feature_names, window_sizes, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, feature_name in enumerate(feature_names):
            plt.figure(figsize=(12, 6))
            for j, (time, features) in enumerate(zip(time_vector, normalizedFeatures)):
                label = f'{window_sizes[j]}-sec window'
                plt.plot(time, features[:, i], label=label, alpha=0.5, marker='o', markersize=4)

            plt.title(f'{feature_name} over Time with Different Window Sizes')
            plt.xlabel('Time (seconds)')
            plt.ylabel(f'Normalized {feature_name}')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            plot_filename = os.path.join(output_folder, f'{feature_name}_over_time.png')
            plt.savefig(plot_filename, dpi=300)
            plt.show()

        print(f"hoowoo, BVP feature plots saved in {output_folder}")
