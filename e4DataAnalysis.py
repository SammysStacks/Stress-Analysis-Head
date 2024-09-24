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

# Plot 3-axis ACC data
plt.figure(figsize=(8, 6))
plt.plot(acc_data['Timestamp'], acc_data['ACC_X'], label='ACC_X')
plt.plot(acc_data['Timestamp'], acc_data['ACC_Y'], label='ACC_Y')
plt.plot(acc_data['Timestamp'], acc_data['ACC_Z'], label='ACC_Z')
plt.title('3-axis Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.show()

# Plot BVP data
plt.figure(figsize=(8, 6))
plt.plot(bvp_data['Timestamp'], bvp_data['BVP'], color='purple')
plt.title('Blood Volume Pulse (BVP)')
plt.xlabel('Time (s)')
plt.ylabel('BVP (AU)')
plt.show()

# Plot GSR data
plt.figure(figsize=(8, 6))
plt.plot(gsr_data['Timestamp'], gsr_data['GSR'], color='orange')
plt.title('Galvanic Skin Response (GSR)')
plt.xlabel('Time (s)')
plt.ylabel('GSR (µS)')
plt.show()

# Plot Temperature data
plt.figure(figsize=(8, 6))
plt.plot(temp_data['Timestamp'], temp_data['Temp'], color='cyan')
plt.title('Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temp (°C)')
plt.show()
