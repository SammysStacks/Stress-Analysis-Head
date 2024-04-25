#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:04:13 2023

@author: jadelynndaomac
"""

import numpy as np
import pandas as pd
import matplotlib as plt

deg = 3
offset_1 = None



# Voltages
x = []

# Serial Reads
y = []

coeff_1 = np.polyfit(x, y, deg)
func_1 = np.poly1d(coeff_1)

xp = np.linspace(0, 3.3, 100)
_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
(-2, 2)
plt.show()