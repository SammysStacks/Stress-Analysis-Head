import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
import antropy as ant


# Calculate RMSSD
""" RMSSD is a measure of HRV, which reflects the balance between sympathetic and parasympathetic nervous systems. High rmssd usually indicates good autonomic nervous system function and recovery 
    while low vairability may indicate stress, fatigue, or other health issues"""
def rmssd(signal):
    return np.sqrt(np.mean(np.square(np.diff(signal))))

# Calculate pulse amplitude (difference between systolic and diastolic peaks)
""" The amplitude reflects the strength of the pulse wave. Higher pulse amplitude can indicate increased blood pressure or good vascular health, maybe this could also be useful for stress analysis?"""
def pulse_amplitude(systolic_peaks, diastolic_peaks):
    return systolic_peaks - diastolic_peaks

# Calculate pulse width (time duration between systolic and diastolic peaks)
""" Pulse width is linked to arterial stiffness. A wider pulse indicates better vascular compliance and health, while a narrow pulse width may indicate arterial stiffness and poor vascular health"""
def pulse_width(time, systolic_peaks, diastolic_peaks):
    return time[systolic_peaks] - time[diastolic_peaks]

# Data loading
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

excel_data = pd.read_excel(excel_file, sheet_name='BVP')

bvp_signal = excel_data['BVP'].values
time = excel_data['Timestamp'].values

# using scipy library to find peaks
""" Systolic peaks are associated with pressure during contraction; Diastolic peaks are associated with pressure when the heart is at rest between beats"""
systolic_peaks, _ = find_peaks(bvp_signal, distance=30)
diastolic_peaks, _ = find_peaks(-bvp_signal, distance=30)

# Time-domain features
heart_rate = len(systolic_peaks) / (time[-1] - time[0]) * 60  # Heart rate in beats per minute
rmssd_value = rmssd(np.diff(systolic_peaks))  # RMSSD from systolic peaks
pulse_amp = np.mean(pulse_amplitude(bvp_signal[systolic_peaks], bvp_signal[diastolic_peaks]))
pulse_wid = np.mean(pulse_width(time, systolic_peaks, diastolic_peaks))
systolic_value = np.mean(bvp_signal[systolic_peaks])
diastolic_value = np.mean(bvp_signal[diastolic_peaks])

# Frequency-domain features (PSD)
fs = 1/0.0156300067901611
freqs, psd = welch(bvp_signal, fs=fs)

# entropy and DFA
"""Entropy measures the complexity and the regularity of the BVP signal. High entropy indicates high randomness, while low entropy indicates more regular movements"""
entropy_value = ant.perm_entropy(bvp_signal)
""" DFA is a method for quantifying the fractal-like scaling behavior of the signal over differnt time scales; DFA is often used in physiological signals to detect long-range correlations and self-similar patterns
    This helps understand the physiological control mechanisms. For example, a healthy heart display fractal behavior, and deviations from this indicate illness """
dfa_value = ant.detrended_fluctuation(bvp_signal)

# Statistical features
mean_bvp = np.mean(bvp_signal)
"""variance can reflect physical acitivity, emotional stress, or abnormal cardiovascular function"""
variance_bvp = np.var(bvp_signal)
"""Skenewss can help in detecting irregularities in BVP pulseform """
skewness_bvp = skew(bvp_signal)
"""Kurtosis indicate the presense of extreme values or outliers in BVP signals, Higher kurtosis could indicate abrupt, sharp changes in BVP signal (such as sudden shifts in blood pressure)
    which could be a sign of irregular cardia function or arrhythmias"""
kurtosis_bvp = kurtosis(bvp_signal)

# feature compilation
features = {
    "Feature": ["Heart Rate", "RMSSD", "Pulse Amplitude", "Pulse Width",
                "Systolic Peaks", "Diastolic Peaks",
                "Entropy", "DFA", "Mean", "Variance", "Skewness", "Kurtosis"],
    "Value": [heart_rate, rmssd_value, pulse_amp, pulse_wid, systolic_value,
              diastolic_value, entropy_value, dfa_value,
              mean_bvp, variance_bvp, skewness_bvp, kurtosis_bvp]
}

features_df = pd.DataFrame(features)

# Define saving parameters
new_file_name = os.path.splitext(os.path.basename(excel_file))[0] + "_BVP_features.xlsx"
new_file_path = os.path.join(e4_watch_data_folder, new_file_name)

features_df.to_excel(new_file_path, index=False)

print(f"BVP features saved to {new_file_path}")

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(features["Feature"], features["Value"], color='crimson', alpha=0.7)
plt.title('BVP Feature Extraction')
plt.xlabel('Feature Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
