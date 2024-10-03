import os
import pandas as pd
import matplotlib.pyplot as plt

# User specific details
date = "20240924"
user = "Ruixiao"
experiment = "Baseline"

# Get the directory and specify the saving parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
raw_data_folder = os.path.join(e4_watch_data_folder, "rawData")
experiment_folder = os.path.join(raw_data_folder, f'{date}_{user}_{experiment}')

# Ensure that directories exist
os.makedirs(experiment_folder, exist_ok=True)

# Excel file location
excel_file = os.path.join(e4_watch_data_folder, f'{date}_{user}_E4_{experiment}.xlsx')

# Read in the sheets
excel_data = pd.read_excel(excel_file, sheet_name=None)

# Extract the data separately for plotting purposes
acc_data = excel_data['ACC']
bvp_data = excel_data['BVP']
gsr_data = excel_data['EDA']
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

# Plot 3-axis acceleration
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
acc_plot_filename = os.path.join(experiment_folder, f'{date}_{user}_{experiment}_Acceleration.png')
plt.savefig(acc_plot_filename, dpi=300)
plt.show()

# Plot BVP data
plt.figure()
plt.plot(bvp_data['Timestamp'], bvp_data['BVP'], color='#BA55D3', linestyle='-', alpha=0.85)  # Medium orchid
plt.title('Blood Volume Pulse (BVP)')
plt.xlabel('Time (s)')
plt.ylabel('BVP (AU)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
bvp_plot_filename = os.path.join(experiment_folder, f'{date}_{user}_{experiment}_BVP.png')
plt.savefig(bvp_plot_filename, dpi=300)
plt.show()

# Plot EDA data
plt.figure()
plt.plot(gsr_data['Timestamp'], gsr_data['GSR'], color='#FFA07A', linestyle='-', alpha=0.85)  # Light salmon
plt.title('Galvanic Skin Response (GSR)')
plt.xlabel('Time (s)')
plt.ylabel('GSR (µS)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
gsr_plot_filename = os.path.join(experiment_folder, f'{date}_{user}_{experiment}_GSR.png')
plt.savefig(gsr_plot_filename, dpi=300)
plt.show()

# Plot Temperature data
plt.figure()
plt.plot(temp_data['Timestamp'], temp_data['Temp'], color='#20B2AA', linestyle='-', alpha=0.85)  # Light sea green
plt.title('Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
temp_plot_filename = os.path.join(experiment_folder, f'{date}_{user}_{experiment}_Temp.png')
plt.savefig(temp_plot_filename, dpi=300)
plt.show()
