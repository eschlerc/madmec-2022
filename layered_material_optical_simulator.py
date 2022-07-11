# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:21:09 2022

@author: Chris Eschler
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tmm   # Optical calculations -- "transfer matrix method"

# =============================================================================
# Functions and initialization
# =============================================================================

# Initialize empty dataframe for n,k data
# wl is wavelength for n values, wl.1 is wavelength for k values
nk_data = pd.DataFrame(columns=['material', 'wl', 'n', 'wl.1', 'k'])
nk_interp = {}          # Initialize empty dict for interpolated data
imported_materials = [] # Initialize empty array for imported materials


# List of wavelengths to be used for calculations, in um
lambda_list = np.concatenate((np.arange(0.300, 1.700, 0.002), 
                          np.arange(1.700, 15.001, 0.010)))

degree = np.pi/180  # Easy conversion between degrees and radians


# Quicker way to input relative paths from file location
# Change path in function to the dir containing this file on your system
def rel_path(path):
    return os.path.join('C:/Users/eschl/Dropbox (MIT)/MIT/_Grad/madmec-2022/', path)


# Add n,k data about a given material to nk_data from filename
# csv file needs to have column headers "wl", "n", "wl", and "k"
# All 4 columns must be the same length for the interpolation to work
def import_nk_data(filename, material):
    global nk_data
    global imported_materials
    nk_raw_data = pd.read_csv(rel_path(filename))
    nk_raw_data = nk_raw_data.assign(material=material)
    nk_data = pd.concat([nk_data, nk_raw_data], ignore_index=True)
    imported_materials.append(material)


# Interpolate to all the wavelength values given in lambda_list using
# quadratic interpolation. Can handle n and k lists with different wavelengths
# as input, as long as they have the same number of values.
def interp_nk_data(material):
    # Look only at the data for desired material
    group = nk_data.loc[nk_data['material']==material]
    n_interp = interp1d(group['wl'], group['n'], kind='quadratic', 
                        bounds_error=False, fill_value='extrapolate')
    k_interp = interp1d(group['wl.1'], group['k'], kind='quadratic',
                        bounds_error=False, fill_value='extrapolate')
    # Combine n and k into complex-valued refrac indices at each wavelength
    nk_interp[material] = (np.maximum(0,n_interp(lambda_list)) + 
                           1j*np.maximum(0,k_interp(lambda_list)))


# Plot the n and k at each wavelength of interest for a given material.
# plot_raw_data: whether the points from the original dataset are
# included on the plot. Useful for checking accuracy of interpolation and
# extrapolation. Set False to make plot less crowded.
# xlim: wavelength limits, defaults to full range of lambda_list.
def plot_nk_interp(material, *, plot_raw_data=True, 
                   xlim=[min(lambda_list), max(lambda_list)], **kwargs):
    plt.figure()
    plt.plot(lambda_list, nk_interp[material].real, 'b-', label='n')
    plt.plot(lambda_list, nk_interp[material].imag, 'r-', label='k')
    if plot_raw_data:
        plt.plot(nk_data.loc[nk_data['material']==material, ['wl']], 
                 nk_data.loc[nk_data['material']==material, ['n']],'bo')
        plt.plot(nk_data.loc[nk_data['material']==material, ['wl.1']], 
                 nk_data.loc[nk_data['material']==material, ['k']],'ro')
    plt.xlabel('Wavelength ($\mu$m)')
    plt.xlim(xlim)
    plt.ylabel('Material n,k')
    plt.legend()
    plt.show()


# Lists are in order of light travel
# th is incoming light angle relative to normal, in degrees!
def plot_atr(material_list, d_list, c_list, th, *,
             xlim=[0, 15], **kwargs):
    # Assumes equally weighted sum of s- and p-polarized light
    T_list = []                         # Initialize transmission list
    R_list = []                         # Initialize reflection list
    for i in range(len(lambda_list)):
        # Refrac index of each layer in the stack, eval'd at current wavelength
        # Maps material_list onto the function for finding n,k
        n_list = list(map(lambda material: nk_interp[material][i], material_list))
        
        # Calculate all optical properties
        s_pol = tmm.inc_tmm('s', n_list, d_list, c_list, th*degree, lambda_list[i])
        p_pol = tmm.inc_tmm('p', n_list, d_list, c_list, th*degree, lambda_list[i])
        
        # Transmission by wavelength
        T_list.append(1/2*s_pol['T'] + 1/2*p_pol['T'])
       
        # Reflection by wavelength
        R_list.append(1/2*s_pol['R'] + 1/2*p_pol['R'])
    
    # A = 1-T-R, calculate absorption
    A_list = np.ones(len(lambda_list))-T_list-R_list
    
    # Plot
    plt.figure()
    plt.plot(lambda_list, T_list, label='Transmission')
    plt.plot(lambda_list, R_list, label='Reflection')
    plt.plot(lambda_list, A_list, label='Absorption')
    plt.xlabel('Wavelength ($\mu$m)')
    plt.xlim(xlim)
    # plt.xscale('log')
    plt.ylabel('Fraction of power')
    plt.ylim([0, 1])
    # plt.title('')
    plt.legend()
    plt.show()

# =============================================================================
# Run
# =============================================================================


import_nk_data('Refractive_indices/nk_PMMA.csv', 'PMMA')
import_nk_data('Refractive_indices/nk_PC.csv', 'PC')
import_nk_data('Refractive_indices/nk_soda-lime-glass.csv', 'Glass')
import_nk_data('Refractive_indices/nk_fused-silica.csv', 'Silica')
import_nk_data('Refractive_indices/nk_ITO.csv', 'ITO')
import_nk_data('Refractive_indices/nk_aWO3.csv', 'a-WO3')
import_nk_data('Refractive_indices/nk_aLi0_06WO3.csv', 'a-Li0.06WO3')
import_nk_data('Refractive_indices/nk_aLi0_18WO3.csv', 'a-Li0.18WO3')
import_nk_data('Refractive_indices/nk_aLi0_34WO3.csv', 'a-Li0.34WO3')
import_nk_data('Refractive_indices/n_air.csv', 'Air')

# Interpolate all the imported materials
for material in imported_materials:
    interp_nk_data(material)

plot_nk_interp('ITO', xlim=[0.3, 1.])
    
# print(min(nk_interp['aWO3'].imag)) # Check min k value, ensure non-negative

material_list = ['Air', 'ITO', 'Glass', 'Air'] # List of materials in order of stack
d_list = [np.inf, 0.1, 400., np.inf]  # Thickness of each layer, in um
c_list = ['i', 'c', 'i', 'i']          # 'c' for coherent, 'i' for incoherent layer
plot_atr(material_list, d_list, c_list, 0)
plot_atr(material_list, d_list, c_list, 0, xlim=[0.3, 1.0]) # Focuses on incoming solar radiation
