import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from scipy import integrate
import antropy as ant

# Calculate RMSSD (Root Mean Square of Successive Differences)
def rmssd(signal):
    return np.sqrt(np.mean(np.square(np.diff(signal))))

# Calculate LF/HF ratio
def lf_hf_ratio(psd, freqs):
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    lf_power = integrate.simps(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], dx=freqs[1] - freqs[0])
    hf_power = integrate.simps(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], dx=freqs[1] - freqs[0])
    return lf_power / hf_power if hf_power != 0 else np.nan

# Calculate pulse amplitude (difference between systolic and diastolic peaks)
def pulse_amplitude(systolic_peaks, diastolic_peaks):
    return systolic_peaks - diastolic_peaks

# Calculate pulse width (time duration between systolic and diastolic peaks)
def pulse_width(time, systolic_peaks, diastolic_peaks):
    return time[systolic_peaks] - time[diastolic_peaks]

# Load the BVP data from the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

# Read the Excel file (assuming BVP sheet)
excel_data = pd.read_excel(excel_file, sheet_name='BVP')

# Extract BVP data and time
bvp_signal = excel_data['BVP'].values
time = excel_data['Timestamp'].values

# Find systolic and diastolic peaks using scipy's find_peaks
systolic_peaks, _ = find_peaks(bvp_signal, distance=30)  # Adjust distance for heart rate
diastolic_peaks, _ = find_peaks(-bvp_signal, distance=30)

# Time-domain features
heart_rate = len(systolic_peaks) / (time[-1] - time[0]) * 60  # Heart rate in beats per minute
rmssd_value = rmssd(np.diff(systolic_peaks))  # RMSSD from systolic peaks
pulse_amp = np.mean(pulse_amplitude(bvp_signal[systolic_peaks], bvp_signal[diastolic_peaks]))
pulse_wid = np.mean(pulse_width(time, systolic_peaks, diastolic_peaks))
systolic_value = np.mean(bvp_signal[systolic_peaks])
diastolic_value = np.mean(bvp_signal[diastolic_peaks])

# Frequency-domain features (PSD and LF/HF ratio)
fs = 50  # Sampling frequency (adjust if needed)
freqs, psd = welch(bvp_signal, fs=fs)
lf_hf_ratio_value = lf_hf_ratio(psd, freqs)

# Non-linear features (entropy and DFA)
entropy_value = ant.perm_entropy(bvp_signal)
dfa_value = ant.detrended_fluctuation(bvp_signal)

# Statistical features
mean_bvp = np.mean(bvp_signal)
variance_bvp = np.var(bvp_signal)
skewness_bvp = skew(bvp_signal)
kurtosis_bvp = kurtosis(bvp_signal)

# Store features in a dictionary
features = {
    "Feature": ["Heart Rate", "RMSSD", "Pulse Amplitude", "Pulse Width",
                "Systolic Peaks", "Diastolic Peaks", "LF/HF Ratio",
                "Entropy", "DFA", "Mean", "Variance", "Skewness", "Kurtosis"],
    "Value": [heart_rate, rmssd_value, pulse_amp, pulse_wid, systolic_value,
              diastolic_value, lf_hf_ratio_value, entropy_value, dfa_value,
              mean_bvp, variance_bvp, skewness_bvp, kurtosis_bvp]
}

# Convert the dictionary to a DataFrame
features_df = pd.DataFrame(features)

# Define new file name with original file name + "BVP features"
new_file_name = os.path.splitext(os.path.basename(excel_file))[0] + "_BVP_features.xlsx"
new_file_path = os.path.join(e4_watch_data_folder, new_file_name)

# Save the extracted features to the new Excel file
features_df.to_excel(new_file_path, index=False)

print(f"BVP features saved to {new_file_path}")

# Plot the features
plt.figure(figsize=(10, 6))
plt.barh(features["Feature"], features["Value"], color='steelblue', alpha=0.7)
plt.title('BVP Feature Extraction')
plt.xlabel('Feature Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
