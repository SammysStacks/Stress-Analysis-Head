\import os
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
