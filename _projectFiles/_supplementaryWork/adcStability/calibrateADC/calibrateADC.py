#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:39:29 2022

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ----------------------------- Import Modules ----------------------------- #

# Basic Modules
import sys
import numpy as np
# Plotting Modules
import matplotlib.pyplot as plt

# Import Header Folder
sys.path.append('../../')
import excelProcessing_Supp as excelDataProtocol       # Functions to Save/Read in Data from Excel

# -------------------------------------------------------------------------- #
# ------------------------ Extract Calibration Data ------------------------ #

# Specify file information
testSheetNum = 0
calibrationFile = 'calibrationADCData.xlsx'

# Extract the calibration data
voltages, serialReads = excelDataProtocol.getExcelData().getData(calibrationFile, testSheetNum = testSheetNum)[0]

# Remove top points
voltages = voltages[1:-2]
serialReads = serialReads[1:-2]
# -------------------------------------------------------------------------- #
# -------------------------- Fit Calibration Data -------------------------- #

# indexes of bounds
cutoff_1 = 7
cutoff_2 = 23
overlap = 2

degrees = 4
# offset = None

# Interval 1: beginning to first cutoff. Get coefficients of fitted function
coeffs_1 = np.polyfit(serialReads[:cutoff_1+overlap], voltages[:cutoff_1+overlap], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.
calibratedVoltages = np.polyval(coeffs_1, serialReads[:cutoff_1])

# Interval 2: first cutoff to second cutoff Get coefficients of fitted function
coeffs_2 = np.polyfit(serialReads[cutoff_1-overlap:cutoff_2+overlap], voltages[cutoff_1-overlap:cutoff_2+overlap], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.

calibratedVoltages = np.append(calibratedVoltages, np.polyval(coeffs_2, serialReads[cutoff_1:cutoff_2]))

# Interval 3: second cutoff to end Get coefficients of fitted function
coeffs_3 = np.polyfit(serialReads[cutoff_2-overlap:], voltages[cutoff_2-overlap:], degrees)
# Use fitted function to calculate the calibrated voltage values, given the serial read value.
calibratedVoltages = np.append(calibratedVoltages, np.polyval(coeffs_3, serialReads[cutoff_2:]))

# percentage difference between the calibrated voltages to what the voltages should actually be at the serial read value
errors = (100*np.subtract(calibratedVoltages, voltages)/voltages)


# -------------------------------------------------------------------------- #
# -------------------------- Plot Calibration Fit -------------------------- #

mainFig = plt.figure()

plt.plot(serialReads, voltages, color= 'red', linewidth=2)

plt.plot(serialReads, calibratedVoltages, color= 'blue', linewidth=2)
plt.plot(serialReads[cutoff_1], calibratedVoltages[cutoff_1], marker='o', color='orange')
plt.plot(serialReads[cutoff_2], calibratedVoltages[cutoff_2], marker='o', color='orange')


plt.title("ADC Calibration")
plt.xlabel("Serial Reading (AU)")
plt.ylabel("Voltage (Volts)")
plt.legend(["Raw Readings", "Calibrated Readings", "Interval bounds"])

errorFig = mainFig.add_axes([0.6, 0.25, 0.25, 0.25])  #[lowerCorner_x, lowerCorner_y, width, height]
errorFig.plot(serialReads, errors, color = 'green')

errorFig.set_title("Calibration Error")
errorFig.set_xlabel("Serial Reading")
errorFig.set_ylabel("Voltage Error (%)")
errorFig.set_ylim(-1, 1)

plt.show()

# -------------------------------------------------------------------------- #




