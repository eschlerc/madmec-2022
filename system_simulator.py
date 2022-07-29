# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:10:42 2022

@author: Chris Eschler
"""

import os
import sys
sys.path.append('C:/Users/eschl/Dropbox (MIT)/MIT/_Grad/madmec-2022')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# import spectra as spec

# =============================================================================
# Inputs
# =============================================================================

k_gr = 2 # Thermal conductivity of soil, W/m/K


# =============================================================================
# Simulation variables
# =============================================================================

time_step = 1 # 1 hour
hours_per_year = 8760

# Soil variables
dz_small = 0.1 # Thickness (m) of thinner surface soil layers
dz_large = 1. # Thickness (m) of thicker deep soil layers
z_changeover = 1. # Depth at which dz switches from dz_small to dz_large
z_const_temp = 10. # Depth at which soil temp is considered constant

# List of depths for which temperatures will be tracked/simulated
z_list = np.concatenate((np.arange(0, z_changeover, dz_small),
                         np.arange(z_changeover, z_const_temp, dz_large)))
# List of layer thicknesses based on z_list
dz_list = np.concatenate((dz_small*np.ones(int(z_changeover/dz_small)), 
                          dz_large*np.ones(int((z_const_temp-z_changeover)/dz_large))))

data = [] # Initialize empty list for all data

for i in range(100):
    x = i
    y = i**2
    z = i%6
    d = {'x': x, 'y': y, 'z': z}
    data.append(d)
df = pd.DataFrame(data)
