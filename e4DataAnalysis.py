import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the Excel file in the e4WatchData folder
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
experimental_data_folder = os.path.join(script_dir, "_experimentalData")
e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
excel_file = os.path.join(e4_watch_data_folder, "E4_data.xlsx")

# Read the Excel file
excel_data = pd.read_excel(excel_file, sheet_name=None)  # sheet_name=None reads all sheets

# Extract data from each sheet
acc_data = excel_data['ACC']  # ACC sheet
bvp_data = excel_data['BVP']  # BVP sheet
gsr_data = excel_data['GSR']  # GSR sheet
temp_data = excel_data['Temp']  # Temp sheet

# Plotting the data
plt.figure(figsize=(12, 10))

# Plot ACC data (3-axis acceleration)
plt.subplot(4, 1, 1)
plt.plot(acc_data['Timestamp'], acc_data['ACC_X'], label='ACC_X')
plt.plot(acc_data['Timestamp'], acc_data['ACC_Y'], label='ACC_Y')
plt.plot(acc_data['Timestamp'], acc_data['ACC_Z'], label='ACC_Z')
plt.title('3-axis Acceleration')
plt.ylabel('Acceleration (g)')
plt.legend()

# Plot BVP data
plt.subplot(4, 1, 2)
plt.plot(bvp_data['Timestamp'], bvp_data['BVP'], color='purple')
plt.title('Blood Volume Pulse (BVP)')
plt.ylabel('BVP (AU)')

# Plot GSR data
plt.subplot(4, 1, 3)
plt.plot(gsr_data['Timestamp'], gsr_data['GSR'], color='orange')
plt.title('Galvanic Skin Response (GSR)')
plt.ylabel('GSR (µS)')

# Plot Temperature data
plt.subplot(4, 1, 4)
plt.plot(temp_data['Timestamp'], temp_data['Temp'], color='cyan')
plt.title('Temperature')
plt.ylabel('Temp (°C)')
plt.xlabel('Time (s)')

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
