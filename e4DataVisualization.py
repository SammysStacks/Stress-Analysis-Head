import os
import pandas as pd
import matplotlib.pyplot as plt

# get the directory and specify the saving parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data_baseline_featureAnalysis.xlsx")

# Read in the sheets
excel_data = pd.read_excel(excel_file, sheet_name=None)

# Extract the data separately for plotting purposes
acc_data = excel_data['ACC']
bvp_data = excel_data['BVP']
gsr_data = excel_data['GSR']
temp_data = excel_data['Temp']

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "lines.linewidth": 2.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "figure.figsize": (8, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": True,
    "legend.loc": 'upper right',
    "savefig.format": "png"
})

# plot 3 axis acceleration
plt.figure()
plt.plot(acc_data['Timestamp'], acc_data['ACC_X'], label='ACC_X', color='#6495ED', alpha=0.85)
plt.plot(acc_data['Timestamp'], acc_data['ACC_Y'], label='ACC_Y', color='#66CDAA', alpha=0.85)
plt.plot(acc_data['Timestamp'], acc_data['ACC_Z'], label='ACC_Z', color='#FF6347', alpha=0.85)
plt.title('3-axis Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# plot bvp data
plt.figure()
plt.plot(bvp_data['Timestamp'], bvp_data['BVP'], color='#BA55D3', linestyle='-', alpha=0.85)  # Medium orchid
plt.title('Blood Volume Pulse (BVP)')
plt.xlabel('Time (s)')
plt.ylabel('BVP (AU)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot GSR data
plt.figure()
plt.plot(gsr_data['Timestamp'], gsr_data['GSR'], color='#FFA07A', linestyle='-', alpha=0.85)  # Light salmon
plt.title('Galvanic Skin Response (GSR)')
plt.xlabel('Time (s)')
plt.ylabel('GSR (µS)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Temperature data 
plt.figure()
plt.plot(temp_data['Timestamp'], temp_data['Temp'], color='#20B2AA', linestyle='-', alpha=0.85)  # Light sea green
plt.title('Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

