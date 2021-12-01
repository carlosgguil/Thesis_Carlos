# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/flow_rates_ibs/spatial_maps/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/IBs/spatial_fluxes/'
sys.path.append(folder)

# Change size of figures if wished
FFIG = 0.5
figsize_ = (FFIG*30,FFIG*20)
figsize_4_in_a_row = (FFIG*55,FFIG*15)
figsize_bar = (FFIG*50,FFIG*20)

# rcParams for plots
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 70*FFIG # 80*FFIG 
plt.rcParams['xtick.major.pad'] = 20*FFIG
plt.rcParams['ytick.labelsize'] = 70*FFIG # 80*FFIG
plt.rcParams['ytick.major.pad'] = 20*FFIG
plt.rcParams['axes.labelsize']  = 70*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 20*FFIG
plt.rcParams['axes.titlesize']  = 70*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 30*FFIG
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams['text.usetex'] = True


# cases
cases = ['UG75_DX10_x05mm','UG75_DX10_x10mm', 
         'UG75_DX20_x05mm','UG75_DX20_x10mm','UG75_DX20_x15mm',
         'UG100_DX10_x05mm','UG100_DX10_x10mm', 
         'UG100_DX20_x05mm','UG100_DX20_x10mm','UG100_DX20_x15mm']
         

# Define labels and tags
x_label_y_coord   = r'$y~[\mathrm{mm}]$'
y_label_z_coord   = r'$z~[\mathrm{mm}]$'

label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
labels_cases = [label_UG75_DX10 , label_UG75_DX20,
                label_UG100_DX10, label_UG100_DX20]

title_x_05mm = r'$x = 5~\mathrm{mm}$'
title_x_10mm = r'$x = 10~\mathrm{mm}$'
title_x_15mm = r'$x = 15~\mathrm{mm}$'


bar_label_total_flux = r'$Q_l~[\mathrm{mm}^3 ~ \mathrm{s}^{-1}]$'
bar_label_volume_flux = r'$q_l~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'


# Bounds for plots
x_lim_y_coord = (-10,10)
y_lim_z_coord = (0,15)

x_ticks_ = [-10,-5,0,5,10]
y_ticks_ = [0,3,6,9,12,15]

# For the figure size
AR = 20/15
figsize_ = (AR*FFIG*15,FFIG*15)

# Read data with pickle
fluxes_spatial_data_all = folder+'fluxes_spatial_data_all'
with open(fluxes_spatial_data_all, 'rb') as f:
    dy_dict = pickle.load(f)
    dz_dict = pickle.load(f)
    S_probe_dict = pickle.load(f)
    y_val_dict = pickle.load(f)
    z_val_dict = pickle.load(f)
    map_Q_IB_dict = pickle.load(f)
    map_vol_flux_IB_dict = pickle.load(f)
f.close()



#%% Plot stuff

for i in range(len(cases)):
    case = cases[i]
    
    # split string case to define title
    string_split = case.split('_')
    #resolution   = string_split[-2]
    plane_string = string_split[-1]
    if plane_string == 'x05mm':
        title_ = title_x_05mm
    elif plane_string == 'x10mm':
        title_ = title_x_10mm
    elif plane_string == 'x15mm':
        title_ = title_x_15mm
        
    # get data to plot
    y_val = y_val_dict[case]
    z_val = z_val_dict[case]
    map_Q_IB = map_Q_IB_dict[case]
    map_vol_flux_IB = map_vol_flux_IB_dict[case]
    
    # Total flux map
    plt.figure(figsize=figsize_)
    plt.title(title_)
    #contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
    #           levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.xlabel(x_label_y_coord)
    plt.ylabel(y_label_z_coord)
    plt.xlim(x_lim_y_coord)
    plt.ylim(y_lim_z_coord)
    plt.xticks(x_ticks_)
    plt.yticks(y_ticks_)
    plt.contourf(y_val, z_val, map_Q_IB, cmap = 'binary')
    cbar = plt.colorbar(format=r'$%.d$')
    cbar.set_label(bar_label_total_flux)
    plt.tight_layout()
    plt.savefig(folder_manuscript + case+'_total_flux.pdf')
    plt.show()
    plt.close()
    
    # Volume flux map
    plt.figure(figsize=figsize_)
    plt.title(title_)
    #contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
    #           levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.xlabel(x_label_y_coord)
    plt.ylabel(y_label_z_coord)
    plt.xlim(x_lim_y_coord)
    plt.ylim(y_lim_z_coord)
    plt.xticks(x_ticks_)
    plt.yticks(y_ticks_)
    plt.contourf(y_val, z_val, map_vol_flux_IB, cmap = 'binary')
    #cbar = plt.colorbar(format=r'$%.2f$')
    cbar = plt.colorbar(format=r'$%.d$')
    cbar.set_label(bar_label_volume_flux)
    plt.tight_layout()
    plt.savefig(folder_manuscript + case+'_volume_flux.pdf')
    plt.show()
    plt.close()
    