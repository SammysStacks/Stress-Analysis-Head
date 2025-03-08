import numpy as np
import matplotlib.pyplot as plt

from _projectFiles._supplementaryWork import excelProcessing_Supp

# ------------------------ Extract Calibration Data ------------------------ #

# Specify file information
testSheetNum = 0
calibrationFile = 'calibrationADCData.xlsx'

# Extract the calibration data
voltages, serialReads = excelProcessing_Supp.getExcelData().getData(calibrationFile, testSheetNum=testSheetNum)

# Remove top points
voltages = voltages[1:-2]
serialReads = serialReads[1:-2]

# -------------------------- Fit Calibration Data -------------------------- #

# indexes of bounds
cutoff_1 = 7
cutoff_2 = 23
overlap = 2

degrees = 4
# offset = None

# Interval 1: beginning to first cutoff. Get coefficients of fitted function
coeffs_1 = np.polyfit(serialReads[:cutoff_1 + overlap], voltages[:cutoff_1 + overlap], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.
calibratedVoltages = np.polyval(coeffs_1, serialReads[:cutoff_1])

# Interval 2: first cutoff to second cutoff Get coefficients of fitted function
coeffs_2 = np.polyfit(serialReads[cutoff_1 - overlap:cutoff_2 + overlap], voltages[cutoff_1 - overlap:cutoff_2 + overlap], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.

calibratedVoltages = np.append(calibratedVoltages, np.polyval(coeffs_2, serialReads[cutoff_1:cutoff_2]))

# Interval 3: second cutoff to end Get coefficients of fitted function
coeffs_3 = np.polyfit(serialReads[cutoff_2 - overlap:], voltages[cutoff_2 - overlap:], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.
calibratedVoltages = np.append(calibratedVoltages, np.polyval(coeffs_3, serialReads[cutoff_2:]))

# percentage difference between the calibrated voltages to what the voltages should actually be at the serial read value
errors = (100 * np.subtract(calibratedVoltages, voltages) / voltages)

# -------------------------- Plot Calibration Fit -------------------------- #

# Create main figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot raw and calibrated voltages
ax.plot(serialReads, voltages, color='red', linewidth=2, label="Raw Readings")
ax.plot(serialReads, calibratedVoltages, color='blue', linewidth=2, label="Calibrated Readings")
ax.plot(serialReads[cutoff_1], calibratedVoltages[cutoff_1], marker='o', color='orange', label="Interval bounds")
ax.plot(serialReads[cutoff_2], calibratedVoltages[cutoff_2], marker='o', color='orange')

# Formatting
ax.set_title("ADC Calibration", fontsize=14, fontweight='bold')
ax.set_xlabel("Serial Reading (au)", fontsize=12)
ax.set_ylabel("Voltage (Volts)", fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# Inset plot for calibration error
error_ax = fig.add_axes([0.6, 0.25, 0.25, 0.25])
error_ax.plot(serialReads, errors, color='green')

# Formatting for inset plot
error_ax.set_title("Calibration Error", fontsize=10, fontweight='bold')
error_ax.set_xlabel("Serial Reading", fontsize=8)
error_ax.set_ylabel("Voltage Error (%)", fontsize=8)
error_ax.set_ylim(-1, 1)
error_ax.grid(True, linestyle='--', alpha=0.7)

# Display the plot
fig.savefig("adcCalibration.pdf", bbox_inches='tight', dpi=300)
plt.show()


# -------------------------------------------------------------------------- #
