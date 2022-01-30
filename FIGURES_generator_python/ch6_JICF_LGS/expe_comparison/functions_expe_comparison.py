
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""
import numpy as np

def average_along_y(grid):
    NP_y = grid.grid_size[0]
    NP_z = grid.grid_size[1]
    z_locations   = np.zeros(NP_z)
    vol_flux_along_z = np.zeros(NP_z)
    SMD_along_z = np.zeros(NP_z)
    for m in range(NP_z):
        ly = 0
        vol_flux_x_dy = 0
        SMD_x_dy = 0
        for n in range(NP_y):
            vol_flux_current = grid.map_vol_flux.data[m][n]
            SMD_current = grid.map_SMD.data[m][n]
            if np.isnan(vol_flux_current) or np.isnan(SMD_current):
                continue
            vol_flux_x_dy += vol_flux_current*grid.dy
            SMD_x_dy += SMD_current*vol_flux_current*grid.dy
            ly += grid.dy
          
        z_locations[m] = grid.zz_center[m][0]
        vol_flux_along_z[m] = vol_flux_x_dy/ly
        SMD_along_z[m] = SMD_x_dy/ly/vol_flux_along_z[m]
    
    #vol_flux_along_z = normalize(z_locations, vol_flux_along_z)
    
    return z_locations, vol_flux_along_z*100, SMD_along_z


def average_along_z(grid):
    NP_y = grid.grid_size[0]
    NP_z = grid.grid_size[1]
    y_locations   = np.zeros(NP_y)
    vol_flux_along_y = np.zeros(NP_y)
    SMD_along_y = np.zeros(NP_y)
    for n in range(NP_y):
        lz = 0
        vol_flux_x_dz = 0
        SMD_x_dz = 0
        for m in range(NP_z):
            vol_flux_current = grid.map_vol_flux.data[m][n]
            SMD_current = grid.map_SMD.data[m][n]
            if np.isnan(vol_flux_current) or np.isnan(SMD_current):
                continue
            vol_flux_x_dz += vol_flux_current*grid.dz
            SMD_x_dz += SMD_current*vol_flux_current*grid.dz
            lz += grid.dz
          
        y_locations[n] = grid.yy_center[0][n]
        vol_flux_along_y[n] = vol_flux_x_dz/lz
        SMD_along_y[n] = SMD_x_dz/lz/vol_flux_along_y[n]
    
    #vol_flux_along_y = normalize(y_locations, vol_flux_along_zy)
    
    return y_locations, vol_flux_along_y*100, SMD_along_y
            



def lognormal(x, mean, std):

    # If the following error appears, it is due to the design space of the optimization algorithm
    
    f = np.zeros(len(x))
    for i in range(len(x)):
        #f[i] = 1/(x[i]*np.log(sigmag)*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(x[i]/mean)/np.log(sigmag))**2)
        f[i] = 1/(x[i]*std*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(x[i]) - mean)**2/std**2)
    return f



def normalize(x, signal):
    
    A = 0
    for i in range(len(signal)-1):
        area_i = (signal[i] + signal[i+1])*(x[i+1]-x[i])/2
        A += area_i
    
    signal = signal/A
    
    return signal
    
