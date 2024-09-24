import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy import integrate

# calculate zero-crossing rate
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

# calculate entropy
def entropy(signal):
    value, counts = np.unique(signal, return_counts=True)
    return -np.sum(counts / len(signal) * np.log2(counts / len(signal)))

# calculate band energy in the 2-3 Hz range
def band_energy(signal, fs=50, band=(2, 3)):
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    band_power = integrate.simps(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# calculate spectral flux
def spectral_flux(signal, fs=50):
    freqs, Pxx = welch(signal, fs=fs, nperseg=1024)
    flux = np.sqrt(np.mean(np.diff(Pxx)**2))
    return flux

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name='ACC')  # Only using ACC sheet

# Extract X, Y, Z axis data
acc_x = excel_data['ACC_X'].values
acc_y = excel_data['ACC_Y'].values
acc_z = excel_data['ACC_Z'].values

# Calculate the combined magnitude of the 3 axes
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Time-domain features
mean_acc = np.mean(acc_magnitude)
std_acc = np.std(acc_magnitude)
corr_xy = np.corrcoef(acc_x, acc_y)[0, 1]
corr_xz = np.corrcoef(acc_x, acc_z)[0, 1]
corr_yz = np.corrcoef(acc_y, acc_z)[0, 1]
kurtosis_acc = kurtosis(acc_magnitude)
skewness_acc = skew(acc_magnitude)
zcr_acc = zero_crossing_rate(acc_magnitude)
entropy_acc = entropy(acc_magnitude)

# Frequency-domain features
band_energy_acc = band_energy(acc_magnitude)
spectral_flux_acc = spectral_flux(acc_magnitude)

# Store features in a dictionary
features = {
    "Feature": ["Mean", "Standard Deviation", "Correlation (X-Y)", "Correlation (X-Z)",
                "Correlation (Y-Z)", "Kurtosis", "Skewness", "Zero Crossing Rate",
                "Entropy", "Band Energy (2-3 Hz)", "Spectral Flux"],
    "Value": [mean_acc, std_acc, corr_xy, corr_xz, corr_yz, kurtosis_acc, skewness_acc,
              zcr_acc, entropy_acc, band_energy_acc, spectral_flux_acc]
}

# Convert the dictionary to a DataFrame
features_df = pd.DataFrame(features)

# Define new file name with original file name + "3_axis_acceleration_feature"
new_file_name = os.path.splitext(os.path.basename(excel_file))[0] + "_3_axis_acceleration_feature.xlsx"
new_file_path = os.path.join(e4_watch_data_folder, new_file_name)

# Save the extracted features to the new Excel file
features_df.to_excel(new_file_path, index=False)

print(f"Features saved to {new_file_path}")

# Plot the features
plt.figure(figsize=(10, 6))
plt.barh(features["Feature"], features["Value"], color='steelblue', alpha=0.7)
plt.title('Feature Extraction from 3-axis Acceleration Data')
plt.xlabel('Feature Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
