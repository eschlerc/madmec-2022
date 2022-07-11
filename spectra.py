# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:59:12 2022

@author: Chris Eschler
"""

import os
import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt

# =============================================================================
# Constants
# =============================================================================
h = 6.6261e-34      # J*s
c = 2.9979e14       # micron/s
kb = 1.3806e-23     # J/K
nAv = 6.0221e23     # Avogadro's number


# =============================================================================
# Functions and initialization
# =============================================================================
# Quicker way to input relative paths from file location
# Change path in function to the dir containing this file on your system
def rel_path(path):
    return os.path.join('C:/Users/eschl/Dropbox (MIT)/MIT/_Grad/madmec-2022/', path)


# =============================================================================
# Data imports
# =============================================================================
# AM1.5 global solar irradiance, W*m^-2*nm^-1
am15 = pd.read_csv(rel_path('Insolation/AM1_5.csv'), 
                     header=0, index_col=0).squeeze('columns')

am15_integral = np.trapz(am15, am15.index) # Approx. integral of AM1.5, W/m^2

plt.figure()
plt.plot(am15)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Global irradiance (W/m$^2$/$\mu$m)')

