import os
import numpy as np
import scipy
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


class accelerationProtocol:
    def __init__(self, signalX, signalY, signalZ, samplingFrequency, windowSizes, overlap=0.25):
        self.signal = self.accelerationMagnitude(signalX, signalY, signalZ)
        self.samplingFrequency = samplingFrequency
        self.windowSizes = windowSizes
        self.overlap = overlap
        # HardCoded Feature Names
        self.featureNames = ['mean', 'std', 'kurtosis', 'skewness', 'zcr', 'entropy', 'band_energy', 'spectral_flux']

    def accelerationMagnitude(self, signalX, signalY, signalZ):
        # direction agnostic analysis
        return np.sqrt(signalX**2 + signalY**2 + signalZ**2)

    @staticmethod
    def normalizeWindowedFeatures(features_all_windows_list):
        # takes in a list of all windowed features
        scaler = MinMaxScaler()
        features_all_windows_normalized = []
        for features in features_all_windows_list:
            features_normalized = scaler.fit_transform(features)
            features_all_windows_normalized.append(features_normalized)
        return features_all_windows_normalized

    # calculate zero-crossing rate
    """this counts how often the signal crosses the zero baseline; used to identify the frequency of small and repetitive movements
        a high zero-crosing rate indicates frequent changes in movement, which could corresponds to restless states, low value 
        could corresponds to more stable positioning"""
    @staticmethod
    def zero_crossing_rate(signal):
        return ((signal[:-1] * signal[1:]) < 0).sum()

    # calculate shannon entropy
    """entropy is a measure of the randomness in the signal; high entropy indicates high randomness, meaning more irregular movements
        low entropy indicates more regular movements"""
    @staticmethod
    def entropy(signal):
        value, counts = np.unique(signal, return_counts=True)
        return -np.sum(counts / len(signal) * np.log2(counts / len(signal)))

    # calculate band energy in the 2-3 Hz range
    """Band energy measures the energy within a specific band of frequencies; in this case, the 2-3 Hz range is used to capture
        the rhythmic compoments of movement. Higher energy indicate consistent, repetitive movements, while low energy indicates
        a more rhythmic movement, like sendnetary behaviors or rest"""
    @staticmethod
    def band_energy(signal, samplingFreq, band=(2,3)):
        f, Pxx = welch(signal, fs=samplingFreq, nperseg=1024) # f: frequency in Hz. Pxx: power spectral density
        band_power = scipy.integrate.simps(Pxx[(f >= band[0]) & (f <= band[1])])
        return band_power

    # calculate spectral flux
    """Spectral flux measures the difference in the seignal frequency content between consecutive time frames, which helps detect 
        changes in the signal's frequency distribution. High spectral flux indicates a high rate of change in the signal's frequency.
        this can be used to detect changes in the movement intensity or patterns over time. High spectral flux indicates a sudden 
        movement, while low spectral flux indicate a more stable movement"""
    @staticmethod
    def spectral_flux(signal, samplingFreq):
        freqs, Pxx = welch(signal, fs=samplingFreq, nperseg=1024)
        flux = np.sqrt(np.mean(np.diff(Pxx) ** 2))
        return flux

    # Calculate features for each window of the signal
    def extract_features(self, signal, samplingFreq):
        """mean captures the general average value of the signal, which can be used to identify the overall intensity of the movement"""
        mean_acc = np.mean(signal)
        """standard deviation captures the variability of the signal, which can be used to identify the consistency of the movement"""
        std_acc = np.std(signal)
        """kurtosis captures the tailedness or extremity of the data distribution. High kurtosis indicates a that there are infrequent but extreme movements, 
           low kurtosis value indicates a more consistent movement"""
        kurtosis_acc = kurtosis(signal)
        """skewness measures the asymmetry of the data distribution. A positive skew means that there are more frequent small movements 
           and fewer large ones, and vice versa"""
        skewness_acc = skew(signal)
        """zero crossing rate indicates how often the signal crosses the zero baseline"""
        zcr_acc = self.zero_crossing_rate(signal)
        """entropy measures the randomness in the signal; high entropy indicates randomness and irregular movements"""
        entropy_acc = self.entropy(signal)
        """band energy captures the rhythmic components of the movement"""
        band_energy_acc = self.band_energy(signal, samplingFreq)
        """spectral flux detects changes in the signal's frequency distribution"""
        spectral_flux_acc = self.spectral_flux(signal, samplingFreq)

        return [mean_acc, std_acc, kurtosis_acc, skewness_acc, zcr_acc, entropy_acc, band_energy_acc, spectral_flux_acc]

    def sliding_windowed_features(self, singleWindowSize, rawSignal, samplingFreq, overlap=0.25):
        windowed_samples = int(singleWindowSize * samplingFreq) # convert window size to samples
        step_size = int(windowed_samples * (1 - overlap)) # step size adjusted for overlaps
        feature_list = []
        for dataPointer in range(0, len(rawSignal) - windowed_samples + 1, step_size):
            windowed_signal = rawSignal[dataPointer:dataPointer + windowed_samples]
            finalFeatures = self.extract_features(windowed_signal, samplingFreq)
            feature_list.append(finalFeatures)
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
            features = self.sliding_windowed_features(window_size, self.signal, self.samplingFrequency, self.overlap)
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

        print(f"woohoo, Feature plots saved in {output_folder}")
