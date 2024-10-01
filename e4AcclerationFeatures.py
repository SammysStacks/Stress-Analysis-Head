import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy import integrate
from sklearn.preprocessing import MinMaxScaler

# feature selection reference: https://doi.org/10.3390/electronics8121461

# User specific details
date = "20240924"
user = "Ruixiao"
experiment = "Baseline"
biomarker = "Acceleration"

# calculate zero-crossing rate
"""this counts how often the signal crosses the zero baseline; used to identify the frequency of small and repetitive movements
    a high zero-crosing rate indicates frequent changes in movement, which could corresponds to restless states, low value 
    could corresponds to more stable positioning"""
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()


# calculate entropy
"""entropy is a measure of the randomness in the signal; high entropy indicates high randomness, meaning more irregular movements
    low entropy indicates more regular movements"""
def entropy(signal):
    value, counts = np.unique(signal, return_counts=True)
    return -np.sum(counts / len(signal) * np.log2(counts / len(signal)))


# calculate band energy in the 2-3 Hz range
"""Band energy measures the energy within a specific band of frequencies; in this case, the 2-3 Hz range is used to capture
    the rhythmic compoments of movement. Higher energy indicate consistent, repetitive movements, while low energy indicates
    a more rhythmic movement, like sendnetary behaviors or rest"""
def band_energy(signal, fs=1 / 0.03125, band=(2, 3)):
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    band_power = integrate.simpson(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power


# calculate spectral flux
"""Spectral flux measures the difference in the seignal frequency content between consecutive time frames, which helps detect 
    changes in the signal's frequency distribution. High spectral flux indicates a high rate of change in the signal's frequency.
    this can be used to detect changes in the movement intensity or patterns over time. High spectral flux indicates a sudden 
    movement, while low spectral flux indicate a more stable movement"""
def spectral_flux(signal, fs=1 / 0.03125):
    freqs, Pxx = welch(signal, fs=fs, nperseg=1024)
    flux = np.sqrt(np.mean(np.diff(Pxx) ** 2))
    return flux


# Calculate features for each window of the signal
def extract_features(signal):
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
    zcr_acc = zero_crossing_rate(signal)
    """entropy measures the randomness in the signal; high entropy indicates randomness and irregular movements"""
    entropy_acc = entropy(signal)
    """band energy captures the rhythmic components of the movement"""
    band_energy_acc = band_energy(signal)
    """spectral flux detects changes in the signal's frequency distribution"""
    spectral_flux_acc = spectral_flux(signal)

    return [mean_acc, std_acc, kurtosis_acc, skewness_acc, zcr_acc, entropy_acc, band_energy_acc, spectral_flux_acc]


# Apply sliding window to extract features
def sliding_window_features(signal, window_size, overlap=0.25, fs=32):

    window_samples = int(window_size * fs)  # Convert window size to samples
    step_size = int(window_samples * (1 - overlap))  # Step size adjusted for overlap

    feature_list = []
    for start in range(0, len(signal) - window_samples + 1, step_size):
        window_signal = signal[start:start + window_samples]
        feature_list.append(extract_features(window_signal))

    return np.asarray(feature_list)



# Get the Excel file storing the data
script_dir = os.path.dirname(os.path.abspath(__file__))
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, f"{date}_{user}_E4_{experiment}.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name='ACC')  # this only focuses on the accleration sheet

# Extract the 3 axis: X, Y, Z axis data
acc_x = excel_data['ACC_X'].values
acc_y = excel_data['ACC_Y'].values
acc_z = excel_data['ACC_Z'].values

# Calculate the combined magnitude of the 3 axes
acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

# Define window sizes in seconds
window_sizes = [5, 10, 30]  # 5 sec, 10 sec, 30 sec

# Apply sliding window for each window size
features_5s = sliding_window_features(acc_magnitude, 5, overlap=0.25)
features_10s = sliding_window_features(acc_magnitude, 10, overlap=0.25)
features_30s = sliding_window_features(acc_magnitude, 30, overlap=0.25)

# Normalize the feature values for each window size
scaler = MinMaxScaler()
features_5s_norm = scaler.fit_transform(features_5s)
features_10s_norm = scaler.fit_transform(features_10s)
features_30s_norm = scaler.fit_transform(features_30s)

# Time vector for x-axis
fs = 32  # 1/0.03125
time_5s = np.arange(0, len(features_5s)) * 5
time_10s = np.arange(0, len(features_10s)) * 10
time_30s = np.arange(0, len(features_30s)) * 30

# Ensure matching lengths for each time window
assert len(time_5s) == len(features_5s_norm)
assert len(time_10s) == len(features_10s_norm)
assert len(time_30s) == len(features_30s_norm)

# Feature names for saving
feature_names = ["Mean", "Standard Deviation", "Kurtosis", "Skewness",
                 "Zero Crossing Rate", "Entropy", "Band Energy (2-3 Hz)", "Spectral Flux"]

# for saving plots
feature_analysis_folder = os.path.join(e4_watch_data_folder, "featureAnalysis")
accelerations_folder = os.path.join(feature_analysis_folder, f"{date}_{user}_{experiment}_{biomarker}_Features")

# check folder existence
if not os.path.exists(feature_analysis_folder):
    os.makedirs(feature_analysis_folder)
if not os.path.exists(accelerations_folder):
    os.makedirs(accelerations_folder)

# plotting
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(12, 6))
    plt.plot(time_5s, features_5s_norm[:, i], label='5-sec window', color='blue', alpha=0.5, marker='o', markersize=4)
    plt.plot(time_10s, features_10s_norm[:, i], label='10-sec window', color='green', alpha=0.5, marker='o', markersize=4)
    plt.plot(time_30s, features_30s_norm[:, i], label='30-sec window', color='red', alpha=0.5, marker='o', markersize=4)
    plt.title(f'{feature_name} over Time with Different Window Sizes')
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Normalized {feature_name}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(accelerations_folder, f'{feature_name}_over_time.png')
    plt.savefig(plot_filename, dpi=300)

    plt.show()

# Save the features to an Excel file
features_combined_5s = pd.DataFrame(features_5s_norm, columns=feature_names)
features_combined_10s = pd.DataFrame(features_10s_norm, columns=feature_names)
features_combined_30s = pd.DataFrame(features_30s_norm, columns=feature_names)

# define the time coloumn
features_combined_5s.insert(0, 'Time_5s', time_5s)
features_combined_10s.insert(0, 'Time_10s', time_10s)
features_combined_30s.insert(0, 'Time_30s', time_30s)

# FeatureExcel savings

# Create directory structure for saving Excel file
feature_analysis_excel_folder = os.path.join(e4_watch_data_folder, "featureAnalysisExcel")
user_feature_excel_data_folder = os.path.join(feature_analysis_excel_folder, f'{date}_{user}_FeatureExcelData')

if not os.path.exists(feature_analysis_excel_folder):
    os.makedirs(feature_analysis_excel_folder)
if not os.path.exists(user_feature_excel_data_folder):
    os.makedirs(user_feature_excel_data_folder)

# Save each window sizes features to separate sheets in Excel
excel_save_path = os.path.join(user_feature_excel_data_folder, f'{biomarker}_{experiment}_normalized_windowed_features.xlsx')
with pd.ExcelWriter(excel_save_path) as writer:
    features_combined_5s.to_excel(writer, sheet_name='5s_Window', index=False)
    features_combined_10s.to_excel(writer, sheet_name='10s_Window', index=False)
    features_combined_30s.to_excel(writer, sheet_name='30s_Window', index=False)

print(f"Windowed and normalized features saved to {excel_save_path}")
