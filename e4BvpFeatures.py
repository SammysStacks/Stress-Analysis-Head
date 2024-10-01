import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
import antropy as ant
from sklearn.preprocessing import MinMaxScaler

# user-specific details
date = "20240924"
user = "Ruixiao"
experiment = "Baseline"
biomarker = "Bvp"


# Calculate RMSSD
def rmssd(signal):
    """ RMSSD is a measure of HRV, which reflects the balance between sympathetic and parasympathetic nervous systems."""
    return np.sqrt(np.mean(np.square(np.diff(signal))))


# Calculate pulse amplitude (difference between systolic and diastolic peaks)
def pulse_amplitude(systolic_peaks, diastolic_peaks):
    """ Ensure systolic and diastolic peaks have the same length by using the shorter array. valid approach?
        systolic peaks are associated with pressure during contraction, diasolic peaks are associated with pressure during relaxation."""
    min_length = min(len(systolic_peaks), len(diastolic_peaks))
    return systolic_peaks[:min_length] - diastolic_peaks[:min_length]

# Calculate pulse width (time duration between systolic and diastolic peaks)
def pulse_width(time, systolic_peaks, diastolic_peaks):
    """ Ensure systolic and diastolic peaks have the same length by using the shorter array. valid approach?"""
    min_length = min(len(systolic_peaks), len(diastolic_peaks))
    return time[systolic_peaks[:min_length]] - time[diastolic_peaks[:min_length]]


# Calculate features for each window of the signal
def extract_features(signal, time, systolic_peaks, diastolic_peaks):
    heart_rate = len(systolic_peaks) / (time[-1] - time[0]) * 60  # Heart rate in bpm
    rmssd_value = rmssd(np.diff(systolic_peaks))
    pulse_amp = np.mean(pulse_amplitude(signal[systolic_peaks], signal[diastolic_peaks]))
    pulse_wid = np.mean(pulse_width(time, systolic_peaks, diastolic_peaks))
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


# sliding window to extract features
def sliding_window_features(signal, time, window_size, overlap=0.25, fs=64):
    window_samples = int(window_size * fs)  # Convert window size to samples
    step_size = int(window_samples * (1 - overlap))  # Step size adjusted for 25% overlap

    feature_list = []
    for start in range(0, len(signal) - window_samples + 1, step_size):
        window_signal = signal[start:start + window_samples]
        window_time = time[start:start + window_samples]

        systolic_peaks, _ = find_peaks(window_signal, distance=30)
        diastolic_peaks, _ = find_peaks(-window_signal, distance=30)

        features = extract_features(window_signal, window_time, systolic_peaks, diastolic_peaks)
        feature_list.append(features)

    return np.asarray(feature_list)



# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, f"{date}_{user}_E4_{experiment}.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name='BVP')

bvp_signal = excel_data['BVP'].values
time = excel_data['Timestamp'].values

# Define window sizes in seconds
window_sizes = [10, 30, 60]  # 10 sec, 30 sec, 60 sec

# Apply sliding windows
features_10s = sliding_window_features(bvp_signal, time, 10, overlap=0.25)
features_30s = sliding_window_features(bvp_signal, time, 30, overlap=0.25)
features_60s = sliding_window_features(bvp_signal, time, 60, overlap=0.25)

# Normalization
scaler = MinMaxScaler()
features_10s_norm = scaler.fit_transform(features_10s)
features_30s_norm = scaler.fit_transform(features_30s)
features_60s_norm = scaler.fit_transform(features_60s)

# Time vector for x-axis
fs = 64  # 1/0.0156300067901611
time_10s = np.arange(0, len(features_10s)) * 10
time_30s = np.arange(0, len(features_30s)) * 30
time_60s = np.arange(0, len(features_60s)) * 60

feature_names = ["Heart Rate", "RMSSD", "Pulse Amplitude", "Pulse Width", "Systolic Peaks", "Diastolic Peaks",
                 "Entropy", "DFA", "Mean", "Variance", "Skewness", "Kurtosis"]

# Saving Plots
feature_analysis_folder = os.path.join(e4_watch_data_folder, "featureAnalysis")
bvp_features_folder = os.path.join(feature_analysis_folder, f"{date}_{user}_{experiment}_{biomarker}_Features")

os.makedirs(bvp_features_folder, exist_ok=True)

# Plotting
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(12, 6))
    plt.plot(time_10s, features_10s_norm[:, i], label='10-sec window', color='blue', alpha=0.5, marker='o', markersize=4)
    plt.plot(time_30s, features_30s_norm[:, i], label='30-sec window', color='green', alpha=0.5, marker='o', markersize=4)
    plt.plot(time_60s, features_60s_norm[:, i], label='60-sec window', color='red', alpha=0.5, marker='o', markersize=4)
    plt.title(f'{feature_name} over Time with Different Window Sizes')
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Normalized {feature_name}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(bvp_features_folder, f'{feature_name}_over_time.png')
    plt.savefig(plot_filename, dpi=300)

    plt.show()

# Saving features
features_combined_10s = pd.DataFrame(features_10s_norm, columns=feature_names)
features_combined_30s = pd.DataFrame(features_30s_norm, columns=feature_names)
features_combined_60s = pd.DataFrame(features_60s_norm, columns=feature_names)

# time column defnition
features_combined_10s.insert(0, 'Time_10s', time_10s)
features_combined_30s.insert(0, 'Time_30s', time_30s)
features_combined_60s.insert(0, 'Time_60s', time_60s)

feature_analysis_excel_folder = os.path.join(e4_watch_data_folder, "featureAnalysisExcel")
user_feature_excel_data_folder = os.path.join(feature_analysis_excel_folder, f'{date}_{user}_FeatureExcelData')

os.makedirs(user_feature_excel_data_folder, exist_ok=True)

# Saving feature excel filea
excel_save_path = os.path.join(user_feature_excel_data_folder, f'{biomarker}_{experiment}_normalized_windowed_features.xlsx')
with pd.ExcelWriter(excel_save_path) as writer:
    features_combined_10s.to_excel(writer, sheet_name='10s_Window', index=False)
    features_combined_30s.to_excel(writer, sheet_name='30s_Window', index=False)
    features_combined_60s.to_excel(writer, sheet_name='60s_Window', index=False)

print(f"BVP features saved to {excel_save_path}")
