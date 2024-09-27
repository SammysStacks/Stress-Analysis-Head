import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy import integrate
from sklearn.preprocessing import MinMaxScaler

from .globalProtocol import globalProtocol

class accelerationProtocol(globalProtocol):

    def __init__(self, windowSize, overlap, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        super().__init__("Acceleration", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.windowSize = windowSize
        self.overlap = overlap



    def zero_crossing_rate(self, signal):
        return ((signal[:-1] * signal[1:]) < 0).sum()

    def entropy(self, signal):
        # TODO: note we already have spectral_entropy in universalProtocol; check implementation and use that one
        value, counts = np.unique(signal, return_counts=True)
        return -np.sum(counts / len(signal) * np.log2(counts / len(signal)))

    def band_energy(self, signal, band=(2, 3)):
        # TODO: note we already have bandpower method in universalProtocol; check implementation and use that one
        fs = self.samplingFreq
        f, Pxx = welch(signal, fs=fs, nperseg=1024)
        band_power = integrate.simpson(Pxx[(f >= band[0]) & (f <= band[1])])
        return band_power

    def spectral_flux(self, signal):
        fs = self.samplingFreq
        freqs, Pxx = welch(signal, fs=fs, nperseg=1024)
        flux = np.sqrt(np.mean(np.diff(Pxx) ** 2))
        return flux

    def extract_features(self, timePoints, signal):
        # ---------------------------------- Data Preprocessing ----------------------------------

        # Normalize the data
        standardized_signal = self.universalMethods.standardizeData(signal)
        if all(standardized_signal == 0):
            return [0 for _ in range(26)]
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
        band_energy_acc = self.band_energy(signal)
        """spectral flux detects changes in the signal's frequency distribution"""
        spectral_flux_acc = self.spectral_flux(signal)

        finalFeatures = []
        finalFeatures.extend([mean_acc, std_acc, kurtosis_acc, skewness_acc, zcr_acc, entropy_acc, band_energy_acc, spectral_flux_acc])

        return finalFeatures

    def sliding_window_features(self, signal):
        window_samples = int(self.windowSize * self.samplingFreq) # convert window size to samples
        step_size = int(window_samples * (1 - self.overlap)) # step size adjusted for overlap

        finalFeatures_windowed = []
        for start in range(0, len(signal) - window_samples + 1, step_size):
            window_signal = signal[start:start + window_samples]
            window_time = self.timePoints[start:start + window_samples]
            # TODO: doublke check
            features = self.extract_features(window_time, window_signal)
            finalFeatures_windowed.append(features)

        return None
    # Unfinished