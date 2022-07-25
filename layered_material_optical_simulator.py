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
tick_list   = np.concatenate((np.arange(0.3, 1., 0.1), np.arange(1, 16, 1)))
tick_labels = [None, '0.4', None, None, '0.7', None, None, '1', '2', '3', '4',
               '5', None, None, None, None, '10', None, None, None, None, '15']

degree = np.pi/180  # Easy conversion between degrees and radians


# Quicker way to input relative paths from file location
# Change path in function to the dir containing this file on your system
def rel_path(path):
    return os.path.join('C:/Users/eschl/Dropbox (MIT)/MIT/_Grad/madmec-2022/', path)


# Add n,k data about a given material to nk_data from filename
# csv file needs to have column headers "wl", "n", "wl", and "k"
def import_nk_data(filename, material):
    global nk_data, imported_materials
    nk_raw_data = pd.read_csv(rel_path(filename))
    nk_raw_data = nk_raw_data.assign(material=material)
    nk_data = pd.concat([nk_data, nk_raw_data], ignore_index=True)
    imported_materials.append(material)


# Calculate n,k from a function of wavelength (um) and add it to nk_interp
def import_from_eqn(function, material):
    global nk_interp
    nk_interp[material] = function(lambda_list)


# ITO n,k model from Del Villar et al. 2010
def ito_nk_eqn(wl):
    om = 2*np.pi*3e8/(wl*1e-6)
    return(np.sqrt(3.57 - 1.89e15**2/(om**2 + 1j*om/6.34e-15) + 
                   0.49*5.61e15**2/(5.61e15**2 - om**2 - 1j*9.72e13*om)))


# Effective medium approximation based on Maxwell Garnet equation
# Probably a poor approximation, but easy to implement
# Used for small volume fraction of inclusions in a matrix
# matrix_material: string name of matrix material from list of imported materials
# inclusion_material: string name of inclusion material " "
# vol_frac: volume fraction of inclusions
def effective_medium(matrix_material, inclusion_material, vol_frac):
    eps_matrix = nk_interp[matrix_material]**2
    eps_inclusion = nk_interp[inclusion_material]**2
    return np.sqrt(eps_matrix*(2*vol_frac*(eps_inclusion-eps_matrix)+eps_inclusion+2*eps_matrix)/
                  (2*eps_matrix+eps_inclusion-vol_frac*(eps_inclusion-eps_matrix)))


# Interpolate to all the wavelength values given in lambda_list using
# quadratic interpolation. Can handle n and k lists with different wavelengths
# as input.
def interp_nk_data(material):
    # Look only at the data for desired material
    # dropna() allows for processing of different length n and k data
    group = nk_data.loc[nk_data['material']==material]
    n_data = group[['wl', 'n']].dropna()
    k_data = group[['wl.1', 'k']].dropna()
    n_interp = interp1d(n_data['wl'], n_data['n'], kind='quadratic', 
                        bounds_error=False, fill_value='extrapolate')
    k_interp = interp1d(k_data['wl.1'], k_data['k'], kind='quadratic',
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
    plt.xscale('symlog', linthresh=1)
    plt.xticks(tick_list, tick_labels)
    plt.xlim(xlim)
    plt.ylabel('Material n,k')
    plt.legend()
    plt.title(material)
    plt.show()


# Calculates A,T,R at each wavelength of interest for layer stack
# Lists are in order of light travel
# th is incoming light angle relative to normal, in *degrees*
def calc_atr(material_list, d_list, c_list, th):
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
    
    return A_list, T_list, R_list


# Plots results from calc_atr
# Lists are in order of light travel
# th is incoming light angle relative to normal, in *degrees*
def plot_atr(material_list, d_list, c_list, th, *,
             xlim=[min(lambda_list), max(lambda_list)], title=None, **kwargs):
    
    A_list, T_list, R_list = calc_atr(material_list, d_list, c_list, th)
    
    # Plot
    plt.figure()
    plt.plot(lambda_list, T_list, label='Transmission')
    plt.plot(lambda_list, R_list, label='Reflection')
    plt.plot(lambda_list, A_list, label='Absorption')
    plt.xlabel('Wavelength ($\mu$m)')
    plt.xscale('symlog', linthresh=1)
    plt.xticks(tick_list, tick_labels)
    plt.xlim(xlim)
    plt.ylabel('Fraction of power')
    plt.ylim([0, 1])
    plt.legend()
    plt.title(title)
    plt.show()

# =============================================================================
# Run
# =============================================================================

# Import all n,k data
import_nk_data('Refractive_indices/n_air.csv', 'Air')                   # 0.23-14.1 um
import_nk_data('Refractive_indices/nk_PMMA.csv', 'PMMA')                # 0.4-19.94 um  !
import_nk_data('Refractive_indices/nk_PC.csv', 'PC')                    # 0.4-19.94 um  !
import_nk_data('Refractive_indices/nk_soda-lime-glass.csv', 'Glass')    # 0.31-80 um
import_nk_data('Refractive_indices/nk_EagleXG.csv', 'Eagle XG')         # 0.19-1.69 um  !!
import_nk_data('Refractive_indices/nk_fused-silica.csv', 'Silica')      # 0.21-50 um
import_nk_data('Refractive_indices/nk_ITO.csv', 'ITO')                  # 0.25-1 um     !!
import_nk_data('Refractive_indices/nk_TiO2.csv', 'TiO2')                # 0.12-125 um
import_nk_data('Refractive_indices/nk_aWO3.csv', 'a-WO3')               # 0.3-2.5 um    !!
import_nk_data('Refractive_indices/nk_aLi0_06WO3.csv', 'a-Li0.06WO3')   # 0.3-2.5 um    !!
import_nk_data('Refractive_indices/nk_aLi0_18WO3.csv', 'a-Li0.18WO3')   # 0.3-2.5 um    !!
import_nk_data('Refractive_indices/nk_aLi0_34WO3.csv', 'a-Li0.34WO3')   # 0.3-2.5 um    !!
import_nk_data('Refractive_indices/nk_ins-VO2.csv', 'VO2-lowT')         # 0.3-15 um
import_nk_data('Refractive_indices/nk_met-VO2.csv', 'VO2-highT')        # 0.3-15 um
import_nk_data('Refractive_indices/nk_Ag.csv', 'Ag')                    # 0.27-24.9 um
import_from_eqn(ito_nk_eqn, 'ITO_eqn')

# print(nk_data.loc[nk_data['material']=='Ag'])
# Interpolate all the imported materials
for material in imported_materials:
    interp_nk_data(material)


eff = effective_medium('PMMA', 'Ag', 0.1)
eff.min()

plt.figure()
plt.plot(lambda_list, nk_interp['Ag'].real, 'b-', label='n Ag')
plt.plot(lambda_list, nk_interp['Ag'].imag, 'b--', label='k Ag')
plt.plot(lambda_list, nk_interp['PMMA'].real, 'r-', label='n PMMA')
plt.plot(lambda_list, nk_interp['PMMA'].imag, 'r--', label='k PMMA')
plt.plot(lambda_list, eff.real, 'k-', label='n$_{eff}$')
plt.plot(lambda_list, eff.imag, 'k--', label='k$_{eff}$')
plt.xlabel('Wavelength ($\mu$m)')
plt.xlim([0.3, 15])
plt.xscale('symlog', linthresh=1)
plt.xticks(tick_list, tick_labels)
plt.ylabel('Material n,k')
plt.ylim([0, 5])
plt.legend()
plt.title('20% AgNW in PMMA')
plt.show()


plot_nk_interp('Ag')

    
# print(min(nk_interp['aWO3'].imag)) # Check min k value, ensure non-negative

# Set up multilayer stack layout
material_list = ['Air', 'Silica', 'VO2-lowT', 'Eagle XG', 'Air'] # List of materials from outside to inside
d_list = [np.inf, 0.167, 0.04, 700., np.inf]  # Thickness of each layer, in um
c_list = ['i', 'c', 'c', 'i', 'i']          # 'c' for coherent, 'i' for incoherent layer

# Incoming light
plot_atr(material_list, d_list, c_list, 0, title='Incoming light')
# plot_atr(material_list, d_list, c_list, 0, xlim=[0.3, 1.0]) # Focuses on insolation

# Outgoing light, uses [::-1] to reverse lists
plot_atr(material_list[::-1], d_list[::-1], c_list[::-1], 0, title='Outgoing light')
