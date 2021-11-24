# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:38:38 2021

@author: d601630
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Change size of figures 
FFIG = 0.5

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 70*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 70*FFIG # 80*FFIG
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
#rc('text.latex', preamble='\usepackage{color}')



folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch9_lagrangian/expe_maps/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/postprocessing/donnees_expe_Renaud/image_processing_with_matlab/csv_output_files/'

# Maps N levels
N_LEVELS = 51

# Map coordinates
x_min = 10
x_max = 35
y_min = -43.5
y_max = 46.5

DX = x_max - x_min
DY = y_max - y_min

# Define figsize
AR = DX/DY
figsize_ = (FFIG*18*AR,FFIG*16)
figsize_maps_subplots = (5*FFIG*16*AR,FFIG*16)


pad_title_maps = 30

# labels
x_label_ = r'$x~[\mathrm{mm}]$'
y_label_ = r'$y~[\mathrm{mm}]$'

label_SMD = r'$SMD~[\mu \mathrm{m}]$'
label_u_axial = r'$u~[\mathrm{m}~s^{-1}]$'
label_u_vertical = r'$v~[\mathrm{m}~s^{-1}]$'


#%% Read maps

# SMD
df_SMD = pd.read_csv(folder+'SMD_values_processed_matlab.csv')
data_SMD = pd.DataFrame.to_numpy(df_SMD)
# y and z coordinates (pixels numbers)
x_values_SMD = np.linspace(1,len(df_SMD.columns),len(df_SMD.columns))
y_values_SMD = np.linspace(1,len(df_SMD),len(df_SMD))

PX_SMD = x_values_SMD[-1]
PY_SMD = y_values_SMD[-1]
# transformation to physical coordinates
x_values_SMD = DX/2/((PX_SMD-1)/2)*x_values_SMD + x_min - DX/2/((PX_SMD-1)/2)
y_values_SMD = DY/(PY_SMD-1)*y_values_SMD+y_min-DY/(PY_SMD-1)
xx_values_SMD, yy_values_SMD = np.meshgrid(x_values_SMD, y_values_SMD)




# u_axial
df_u_axial = pd.read_csv(folder+'u_axial_values_processed_matlab.csv')
data_u_axial = pd.DataFrame.to_numpy(df_u_axial)
# y and z coordinates (pixels numbers)
x_values_u_axial  = np.linspace(1,len(df_u_axial.columns),len(df_u_axial.columns))
y_values_u_axial  = np.linspace(1,len(df_u_axial),len(df_u_axial))

PX_u_axial  = x_values_u_axial[-1]
PY_u_axial  = y_values_u_axial[-1]
# transformation to physical coordinates
x_values_u_axial = DX/2/((PX_u_axial-1)/2)*x_values_u_axial + x_min - DX/2/((PX_u_axial-1)/2)
y_values_u_axial = DY/(PY_u_axial-1)*y_values_u_axial+y_min-DY/(PY_u_axial-1)
xx_values_u_axial, yy_values_u_axial = np.meshgrid(x_values_u_axial, y_values_u_axial)




# u_vertical
df_u_vertical = pd.read_csv(folder+'u_vertical_values_processed_matlab.csv')
data_u_vertical = pd.DataFrame.to_numpy(df_u_vertical)
# y and z coordinates (pixels numbers)
x_values_u_vertical  = np.linspace(1,len(df_u_vertical.columns),len(df_u_vertical.columns))
y_values_u_vertical  = np.linspace(1,len(df_u_vertical),len(df_u_vertical))

PX_u_vertical  = x_values_u_vertical[-1]
PY_u_vertical  = y_values_u_vertical[-1]
# transformation to physical coordinates
x_values_u_vertical = DX/2/((PX_u_vertical-1)/2)*x_values_u_vertical + x_min - DX/2/((PX_u_vertical-1)/2)
y_values_u_vertical = DY/(PY_u_vertical-1)*y_values_u_vertical+y_min-DY/(PY_u_vertical-1)
xx_values_u_vertical, yy_values_u_vertical = np.meshgrid(x_values_u_vertical, y_values_u_vertical)



# levels calculation
levels_map_SMD = np.linspace(9,34,51)
levels_map_u_axial = np.linspace(-23,53,77)
levels_map_u_vertical = np.linspace(-17,17,34)

#%% Plot individual maps

dpi_ = 120

# SMD
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_SMD, yy_values_SMD, data_SMD, 
                   vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
plt.colorbar(format = '$%d$',ticks=levels_map_SMD[::10])
plt.title(label_SMD, pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_map.pdf',bbox_inches='tight')
plt.savefig(folder_manuscript+'SMD_map.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()



# u axial
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_u_axial, yy_values_u_axial, data_u_axial,
                   vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
plt.colorbar(format = '$%d$',ticks=[-20,0,20,40])
plt.title(label_u_axial, pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
plt.savefig(folder_manuscript+'u_axial_map.pdf',bbox_inches='tight')
plt.savefig(folder_manuscript+'u_axial_map.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()

# u vertical
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_u_vertical, yy_values_u_vertical, data_u_vertical,
                   vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
plt.colorbar(format = '$%d$',ticks=[-10,0,10])
plt.title(label_u_vertical, pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
plt.savefig(folder_manuscript+'u_vertical_map.pdf',bbox_inches='tight')
plt.savefig(folder_manuscript+'u_vertical_map.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()

