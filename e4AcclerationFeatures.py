import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy import integrate

# feature selection reference: https://doi.org/10.3390/electronics8121461

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
def band_energy(signal, fs=1/0.03125, band=(2, 3)):
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    band_power = integrate.simpson(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# calculate spectral flux
"""Spectral flux measures the difference in the seignal frequency content between consecutive time frames, which helps detect 
    changes in the signal's frequency distribution. High spectral flux indicates a high rate of change in the signal's frequency.
    this can be used to detect changes in the movement intensity or patterns over time. High spectral flux indicates a sudden 
    movement, while low spectral flux indicate a more stable movement"""
def spectral_flux(signal, fs=1/0.03125):
    freqs, Pxx = welch(signal, fs=fs, nperseg=1024)
    flux = np.sqrt(np.mean(np.diff(Pxx)**2))
    return flux

# get the excel file storing the data
script_dir = os.path.dirname(os.path.abspath(__file__))
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name='ACC')  # this only focuses on the accleration sheet

# Extract the 3 axis: X, Y, Z axis data
acc_x = excel_data['ACC_X'].values
acc_y = excel_data['ACC_Y'].values
acc_z = excel_data['ACC_Z'].values

# Calculate the combined magnitude of the 3 axes
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Time-domain features analysis
"""mean captures the general average value of the signal, which can be used to identify the overall intensity of the movement"""
mean_acc = np.mean(acc_magnitude)
"""standard deviation captures the variability of the signal, which can be used to identify the consistency of the movement"""
std_acc = np.std(acc_magnitude)
"""correlation captures the relationship between the different axes of the signal, which can be used to identify the a certain type of movements"""
corr_xy = np.corrcoef(acc_x, acc_y)[0, 1]
corr_xz = np.corrcoef(acc_x, acc_z)[0, 1]
corr_yz = np.corrcoef(acc_y, acc_z)[0, 1]
"""kurtosis captures the tailedness or extremity of the data distribution. High kurtosis indicates a that there are infrequent but extreme movements, low kurtosis value indicates a more consistent movement"""
kurtosis_acc = kurtosis(acc_magnitude)
"""skewness measures the asymmetry of the data distribution. a positive skew means that there are more frequent small movements and fewer large ones and vice versa, this feature help differentiate between regular and small
    movements and larger irregular movements"""
skewness_acc = skew(acc_magnitude)
zcr_acc = zero_crossing_rate(acc_magnitude)
entropy_acc = entropy(acc_magnitude)

# Frequency-domain features
band_energy_acc = band_energy(acc_magnitude)
spectral_flux_acc = spectral_flux(acc_magnitude)

features = {
    "Feature": ["Mean", "Standard Deviation", "Correlation (X-Y)", "Correlation (X-Z)",
                "Correlation (Y-Z)", "Kurtosis", "Skewness", "Zero Crossing Rate",
                "Entropy", "Band Energy (2-3 Hz)", "Spectral Flux"],
    "Value": [mean_acc, std_acc, corr_xy, corr_xz, corr_yz, kurtosis_acc, skewness_acc,
              zcr_acc, entropy_acc, band_energy_acc, spectral_flux_acc]
}

features_df = pd.DataFrame(features)

# Specify the saving parameters
new_file_name = os.path.splitext(os.path.basename(excel_file))[0] + "_3_axis_acceleration_feature.xlsx"
new_file_path = os.path.join(e4_watch_data_folder, new_file_name)

features_df.to_excel(new_file_path, index=False)

print(f"Features saved to {new_file_path}")

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(features["Feature"], features["Value"], color='steelblue', alpha=0.7)
plt.title('Feature Extraction from 3-axis Acceleration Data')
plt.xlabel('Feature Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
