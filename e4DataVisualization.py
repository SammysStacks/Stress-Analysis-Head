import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the Excel file in the e4WatchData folder
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name=None)  # sheet_name=None reads all sheets

# Extract data from each sheet
acc_data = excel_data['ACC']  # ACC sheet
bvp_data = excel_data['BVP']  # BVP sheet
gsr_data = excel_data['GSR']  # GSR sheet
temp_data = excel_data['Temp']  # Temp sheet

# Plot settings for scientific publications
plt.rcParams.update({
    "font.size": 14,               # General font size
    "axes.labelsize": 16,          # Font size for axis labels
    "axes.titlesize": 18,          # Font size for plot titles
    "legend.fontsize": 14,         # Font size for legend
    "lines.linewidth": 2.5,        # Line width for plot curves
    "xtick.labelsize": 14,         # Font size for x-tick labels
    "ytick.labelsize": 14,         # Font size for y-tick labels
    "figure.dpi": 300,             # High DPI for publication quality
    "figure.figsize": (8, 6),      # Default figure size
    "axes.grid": True,             # Enable grid
    "grid.alpha": 0.3,             # Grid line transparency
    "legend.frameon": True,        # Box around legend
    "legend.loc": 'upper right',   # Legend position
    "savefig.format": "png"        # Save format (can also be 'pdf' or 'eps')
})

# Plot 3-axis ACC data with slightly more saturated colors
plt.figure()
plt.plot(acc_data['Timestamp'], acc_data['ACC_X'], label='ACC_X', color='#6495ED', alpha=0.85)  # Cornflower blue
plt.plot(acc_data['Timestamp'], acc_data['ACC_Y'], label='ACC_Y', color='#66CDAA', alpha=0.85)  # Medium aquamarine
plt.plot(acc_data['Timestamp'], acc_data['ACC_Z'], label='ACC_Z', color='#FF6347', alpha=0.85)  # Tomato red
plt.title('3-axis Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot BVP data with slightly more saturated color
plt.figure()
plt.plot(bvp_data['Timestamp'], bvp_data['BVP'], color='#BA55D3', linestyle='-', alpha=0.85)  # Medium orchid
plt.title('Blood Volume Pulse (BVP)')
plt.xlabel('Time (s)')
plt.ylabel('BVP (AU)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot GSR data with slightly more saturated color
plt.figure()
plt.plot(gsr_data['Timestamp'], gsr_data['GSR'], color='#FFA07A', linestyle='-', alpha=0.85)  # Light salmon
plt.title('Galvanic Skin Response (GSR)')
plt.xlabel('Time (s)')
plt.ylabel('GSR (µS)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Temperature data with slightly more saturated color
plt.figure()
plt.plot(temp_data['Timestamp'], temp_data['Temp'], color='#20B2AA', linestyle='-', alpha=0.85)  # Light sea green
plt.title('Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.fftpack import fft
from scipy import integrate

# Function to calculate zero-crossing rate
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

# Function to calculate entropy
def entropy(signal):
    value, counts = np.unique(signal, return_counts=True)
    return -np.sum(counts / len(signal) * np.log2(counts / len(signal)))

# Function to calculate band energy in the 2-3 Hz range
def band_energy(signal, fs=50, band=(2, 3)):
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    band_power = integrate.simps(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# Function to calculate spectral flux
def spectral_flux(signal, fs=50):
    freqs, Pxx = welch(signal, fs=fs, nperseg=1024)
    flux = np.sqrt(np.mean(np.diff(Pxx)**2))
    return flux

# Load your acceleration data (assuming it's in 'E4_data.xlsx')
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data.xlsx")

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

# Store features for plotting
features = {
    "Mean": mean_acc,
    "Standard Deviation": std_acc,
    "Correlation (X-Y)": corr_xy,
    "Correlation (X-Z)": corr_xz,
    "Correlation (Y-Z)": corr_yz,
    "Kurtosis": kurtosis_acc,
    "Skewness": skewness_acc,
    "Zero Crossing Rate": zcr_acc,
    "Entropy": entropy_acc,
    "Band Energy (2-3 Hz)": band_energy_acc,
    "Spectral Flux": spectral_flux_acc
}

# Plot the features
plt.figure(figsize=(10, 6))
plt.barh(list(features.keys()), list(features.values()), color='steelblue', alpha=0.7)
plt.title('Feature Extraction from 3-axis Acceleration Data')
plt.xlabel('Feature Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
