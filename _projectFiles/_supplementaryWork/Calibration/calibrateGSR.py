#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:39:29 2022

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ----------------------------- Import Modules ----------------------------- #

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# -------------------------------------------------------------------------- #
# ------------------------ Specify Calibration Data ------------------------ #

V_0 = np.array([0.1647, 0.2172, 0.3205, 0.4773, 0.5136, 0.556 , 0.5786, 0.6028,
       0.6083, 0.6267, 0.691 , 0.7703, 0.8145, 0.864 , 0.9122, 1.0561,
       1.1401, 1.2523, 1.3742, 1.5193, 1.6033, 1.7058, 1.8077, 1.9223,
       2.07  , 2.221 , 2.527 , 2.757 , 3.075 , 3.323 , 3.42  , 3.924 ])


R_0 = [3935.  , 2956.  , 1964.  , 1285.4 , 1187.  , 1088.6 , 1041.9 ,
        995.  ,  985.2 ,  953.2 ,  854.8 ,  756.5 ,  710.  ,  663.8 ,
        622.9 ,  524.8 ,  478.5 ,  426.5 ,  380.5 ,  334.1 ,  311.2 ,
        286.6 ,  264.7 ,  243.2 ,  218.6 ,  196.6 ,  160.6 ,  138.87,
        114.25,   98.13,   92.6 ,   67.85]

V_1_5 = np.array([0.6579, 0.6821, 0.7227, 0.8011, 0.9231, 0.951 , 0.9846, 1.0022,
       1.0213, 1.025 , 1.0421, 1.0926, 1.1552, 1.1902, 1.2296, 1.2748,
       1.3921, 1.4617, 1.5443, 1.6438, 1.7653, 1.831 , 1.9157, 1.998 ,
       2.095 , 2.221 , 2.35  , 2.614 , 2.814 , 3.098 , 3.32  , 3.409 ,
       3.877])

R_1_5 = [4929.  , 3949.  , 2958.  , 1977.  , 1289.  , 1192.2 , 1092.  ,
       1045.8 ,  999.  ,  992.3 ,  953.  ,  853.7 ,  755.6 ,  709.  ,
        662.5 ,  615.  ,  517.2 ,  471.1 ,  424.7 ,  378.3 ,  331.9 ,
        311.2 ,  286.6 ,  264.7 ,  243.2 ,  218.6 ,  196.6 ,  160.6 ,
        138.87,  114.25,   98.13,   92.6 ,   67.85]

# -------------------------------------------------------------------------- #
# ---------------------- Interpolate Calibration Data ---------------------- #

# Specify new Interpolation Region
voltageInterp_0 = np.arange(V_0[0], V_0[-1], 0.001)
voltageInterp_1_5 = np.arange(V_1_5[0], V_1_5[-1], 0.001)
# Fit the splines to the data
spline_0 = UnivariateSpline(V_0, R_0, k=5, s=3)
spline_1_5 = UnivariateSpline(V_1_5, R_1_5, k=5, s=3)
# Interpolate the data
interpR_0 = spline_0(voltageInterp_0)
interpR_1_5 = spline_1_5(voltageInterp_1_5)

# plt.plot(V_0, R_0 - spline_0(V_0))
# plt.plot(V_1_5, R_1_5 - spline_1_5(V_1_5))
# plt.show()


# -------------------------------------------------------------------------- #
# -------------------------- Fit Calibration Data -------------------------- #

if False:
    # Define Fitting Function
    def fittingFunction(x, a, b, c, d, e, f):
        return a*np.exp(b*x) + c*np.log(d*x) + e*x + f
    
    bounds = [[-10E5, -10E3, -10E5, 1, -10E5, -10E5], [10E5, 10E2, 10E5, 10E5, 10E5, 10E5]]
    # Fit the interpolated data
    popt_0, pcov_0 = scipy.optimize.curve_fit(fittingFunction, V_0, np.log(R_0), bounds=bounds, maxfev=50000)
    popt_1_5, pcov_1_5 = scipy.optimize.curve_fit(fittingFunction, V_1_5, np.log(R_1_5), bounds=bounds, maxfev = 50000)
    # Reconstruct the final function
    fitV_0 = np.exp(fittingFunction(voltageInterp_0, *popt_0))
    fitV_1_5 = np.exp(fittingFunction(voltageInterp_1_5, *popt_1_5))

if True:
    offset = 50
    index_0 = np.argmin(abs(voltageInterp_0 - 1))
    index_1_5 = np.argmin(abs(voltageInterp_1_5 - 1.2))
    # Fit the interpolated data
    fitCoeff_0 = np.polyfit(voltageInterp_0[index_0-offset:], np.log(interpR_0[index_0-offset:]), 8)
    fitCoeff_1_5 = np.polyfit(voltageInterp_1_5[index_1_5-offset:], np.log(interpR_1_5[index_1_5-offset:]), 8)
    # Fit the interpolated data
    fitCoeff_0_FIRST = np.polyfit(voltageInterp_0[:index_0+offset], np.log(interpR_0[:index_0+offset]), 8)
    fitCoeff_1_5_FIRST = np.polyfit(voltageInterp_1_5[:index_1_5+offset], np.log(interpR_1_5[:index_1_5+offset]), 8)
    # Reconstruct the final function
    fitV_0 = np.exp(np.polyval(fitCoeff_0, voltageInterp_0))
    fitV_1_5 = np.exp(np.polyval(fitCoeff_1_5, voltageInterp_1_5))
    # Reconstruct the final function
    fitV_0[:index_0] = np.exp(np.polyval(fitCoeff_0_FIRST, voltageInterp_0))[:index_0]
    fitV_1_5[:index_1_5] = np.exp(np.polyval(fitCoeff_1_5_FIRST, voltageInterp_1_5))[:index_1_5] 
    # Extract the final equation
    finalEquation_0 = ''
    R = 0; V = 1.46
    for degree in range(len(fitCoeff_0_FIRST)):
        R += fitCoeff_0_FIRST[degree]*V**degree
        finalEquation_0 += ' + ' + str(fitCoeff_0_FIRST[degree]) + '*pow(adcVolts,' + str(len(fitCoeff_0_FIRST)-1-degree) + ')'
    print(finalEquation_0[3:])
    print(R)


# -------------------------------------------------------------------------- #
# -------------------------- Plot Calibration Fit -------------------------- #

plt.plot(V_1_5, R_1_5, 'k', linewidth=4)
plt.plot(voltageInterp_1_5, interpR_1_5, 'o', c='tab:blue', markersize=2)
plt.plot(voltageInterp_1_5, fitV_1_5, 'tab:red', linewidth=2)
plt.show()

plt.plot(voltageInterp_1_5, abs(fitV_1_5 - interpR_1_5)/interpR_1_5)
# plt.ylim(0,2)
plt.show()

plt.plot(voltageInterp_1_5, abs(fitV_1_5 - interpR_1_5))
plt.ylim(0,2)
plt.show()

# plt.plot(V_1_5, R_1_5, 'k', linewidth=4)
# plt.plot(voltageInterp_1_5, interpR_1_5, 'o', c='tab:blue', markersize=2)
# plt.plot(voltageInterp_1_5, fitV_1_5, 'tab:red', linewidth=2)

# plt.xlim(1, 3)
# plt.ylim(0, 1000)
# plt.show()


plt.plot(V_0, R_0, 'k', linewidth=4)
plt.plot(voltageInterp_0, interpR_0, 'o', c='tab:blue', markersize=2)
plt.plot(voltageInterp_0, fitV_0, 'tab:red', linewidth=2)
plt.show()

plt.plot(voltageInterp_0, abs(fitV_0-interpR_0)/interpR_0)
# plt.ylim(0,2)
plt.show()

sys.exit()







plt.plot(V_0, R_0, 'k', linewidth=4)
plt.plot(voltageInterp_0, interpR_0, 'o', c='tab:blue', markersize=2)
plt.plot(voltageInterp_0, fitV_0, 'tab:red', linewidth=2)
plt.show()

plt.plot(voltageInterp_0, interpR_0 - fitV_0)
plt.show()


plt.plot(V_1_5, R_1_5, 'k', linewidth=4)
plt.plot(voltageInterp_1_5, interpR_1_5, 'o', c='tab:blue', markersize=1)
plt.plot(voltageInterp_1_5, fitV_1_5, 'tab:red', linewidth=2)
plt.show()
sys.exit()


plt.plot(voltageInterp_0, interpR_0, 'o', c='k', markersize=2, label="Resistance Calibration")
plt.plot(voltageInterp_1_5, interpR_1_5, 'o', c='tab:red', markersize=2, label="Resistance Calibration with +V Bias")
plt.title("Adding in voltage bias")
plt.ylabel("Resistance (kOhms)")
plt.xlabel("Analog Voltage")
plt.legend()
plt.show()



plt.figure()
ax = plt.gca()

ax.plot(V_0, R_0, "k", linewidth=2, label="Resistance Calibration");
ax.plot([3.2]*len(R), R, "tab:red", linewidth=2, label="Max ADC Voltage");
# ax.fill_between(R[4:6], V[4:6], color='tab:blue', alpha = 0.15, label="Forehead Resistance");
# ax.fill_between(R[5:9], V[5:9], color='tab:red', alpha = 0.15, label="Hand Resistance");
plt.plot(vNew, fit, 'g--', label="Fit")

plt.title("New GSR Board Readings")
plt.xlabel("Resistance (kOhms)")
plt.ylabel("Analog Voltage")
plt.legend()

# -------------------------------------------------------------------------- #




