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
c = 2.9979e8        # m/s
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
# AM1.5 global solar irradiance, normalized to am15_integral
am15 = pd.read_csv(rel_path('Insolation/AM1_5.csv'), 
                     header=0, index_col=0).squeeze('columns')

am15_integral = np.trapz(am15, am15.index) # Approx. integral of AM1.5

plt.figure()
plt.plot(am15)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Global irradiance (W/m$^2$/$\mu$m)')


# Atmospheric transmittance by wavelength
atm_trans = pd.concat([
    pd.read_csv(rel_path('Insolation/cptrans_zm_23_15.txt'),
                header=None, names=['Wavelength (um)', 'Transmittance'],
                sep='\s+'),
    pd.read_csv(rel_path('Insolation/cptrans_nq_23_15.txt'),
                header=None, names=['Wavelength (um)', 'Transmittance'],
                sep='\s+')
    ], ignore_index=True)

plt.figure()
plt.plot(atm_trans['Wavelength (um)'], atm_trans['Transmittance'])
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Atmospheric transmittance')


# Blackbody radiation from Planck's equation in W*m^-2*um^-1
# temp in K, wl in um
def planck(temp, wl):
    global h, c, kb, nAv
    return 1e-6*2*np.pi*h*c**2/(((wl*1e-6)**5)*(np.exp(h*c/(wl*1e-6*kb*temp))-1))

# List of wavelengths to be used for calculations, in um
lambda_list = np.concatenate((np.arange(0.300, 1.700, 0.002), 
                          np.arange(1.700, 100.001, 0.010)))

plt.figure()
plt.plot(lambda_list, list(map(lambda wl: planck(300, wl), lambda_list)))
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Radiant exitance (W/m$^2$/$\mu$m)')
