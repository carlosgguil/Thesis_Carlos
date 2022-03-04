# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:22:20 2021

@author: d601630
"""


#from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.fftpack import fftfreq


FFIG = 0.5
'''
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 60*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'best'
plt.rcParams['text.usetex'] = True
plt.rcParams['legend.framealpha'] = 1.0
'''

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 45*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]
#rc('text.latex', preamble='\usepackage{color}')

figsize_ = (FFIG*30,FFIG*16)

d_inj = 0.45
T     = 1.5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/turbulent_structures/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/turbulence_state_u_mean/'



#%%  cases and labels

cases = [folder + 'DX07p5_lines/',
         folder + 'DX10p0_lines/',
         folder + 'DX15p0_lines/',
         folder + 'lines_u_mean_ics/']

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x ~[\mathrm{mm}]$'
label_z_ax   = '$z ~[\mathrm{mm}]$'

labels_cases = [r'$\mathrm{DX07}$' ,r'$\mathrm{DX10}$',
                r'$\mathrm{DX15}$',r'$\mathrm{No~jet}$']


labels_x_planes = [r'$x/d_\mathrm{inj} = 3.33$',  
                   r'$x/d_\mathrm{inj} = 5.00$',  
                   r'$x/d_\mathrm{inj} = 6.67$']

labels_z_planes = [r'$z = 0.15~\mathrm{mm}$',  
                   r'$z = 0.3~\mathrm{mm}$', 
                   r'$z = 0.6~\mathrm{mm}$', 
                   r'$z = 1~\mathrm{mm}$',  
                   r'$z = 1.5~\mathrm{mm}$'] 


y_coord_lim = (-3,3)
y_coord_ticks = [-2,-1,0,1,2]
u_lim = (-20,60)#(30,120)
u_ticks = [0,30, 60]

#y_ticks_u_vs_x = [-20, 0, 20, 40, 60, 80, 100, 120]

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_y_ax   = '$y ~[\mathrm{mm}]$'

# For z locations
k_low = 4; k_high = 1
#k_low = 2; k_high = -3

label_DX07_low  = 'k'
label_DX07_high = '--k'
label_DX10_low  = 'b'
label_DX10_high = '--b'
label_DX15_low  = 'r'
label_DX15_high = '--r'
label_ics_low  = ':k'
label_ics_high = '--r'

#%% get data

# Define arrays
y_values      = [ [] for m in range(len(cases)) ]
u_mean_values = [ [] for m in range(len(cases)) ]
v_mean_values = [ [] for m in range(len(cases)) ]
w_mean_values = [ [] for m in range(len(cases)) ]

for i in range(len(cases)):
    
    # x planes
    case_i_xD03p33 = cases[i]+'plane_xD_03p33/'
    case_i_xD05p00  = cases[i]+'plane_xD_05p00/'
    case_i_xD06p67   = cases[i]+'plane_xD_06p67/'
    
    # put all folders together
    cases_x_planes = [case_i_xD03p33, 
                      case_i_xD05p00, 
                      case_i_xD06p67]
    names_planes = ['planeXD03p33', 'planeXD05p00', 'planeXD06p67']
    
    # add arrays per each plane
    y_values[i]      = [ [] for m in range(len(cases_x_planes)) ]
    u_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    v_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    w_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    for j in range(len(cases_x_planes)):
        
        # z lines
        line_z0p15 = cases_x_planes[j]+'z_0p15mm.csv'
        line_z0p3 = cases_x_planes[j]+'z_0p3mm.csv'
        line_z0p6   = cases_x_planes[j]+'z_0p6mm.csv'
        line_z1p0 = cases_x_planes[j]+'z_1p0mm.csv'
        line_z1p5   = cases_x_planes[j]+'z_1p5mm.csv'
        line_z2p0   = cases_x_planes[j]+'z_2p0mm.csv'
        
        lines = [line_z0p15, line_z0p3, line_z0p6, 
                 line_z1p0, line_z1p5, line_z2p0]
                 
        
        # add arrays per each plane
        y_values[i][j]      = [ [] for m in range(len(lines)) ]
        u_mean_values[i][j] = [ [] for m in range(len(lines)) ]
        v_mean_values[i][j] = [ [] for m in range(len(lines)) ]
        w_mean_values[i][j] = [ [] for m in range(len(lines)) ]
        
        for k in range(len(lines)):
            df = pd.read_csv(lines[k])
            y_values[i][j][k]      = df['Points_1'].values*1e3
            u_mean_values[i][j][k] = df['U_MEAN_0'].values
            v_mean_values[i][j][k] = df['U_MEAN_1'].values
            w_mean_values[i][j][k] = df['U_MEAN_2'].values
            
            
            



#%% Plots 


u_to_plot = u_mean_values

j = 0  # plane xD = 3.33
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
i = 0  # DX07
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX07_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX07_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 1  # DX10
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX10_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX10_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 2  # DX15
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX15_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX15_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i =  3  # ics, no jet
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_ics_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_ics_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG100_x05.pdf')
plt.show()
plt.close()

j = 1  # plane xD = 5.00
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
i = 0  # DX07
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX07_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX07_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 1  # DX10
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX10_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX10_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 2  # DX15
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX15_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX15_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i =  3  # ics, no jet
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_ics_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_ics_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG100_x05.pdf')
plt.show()
plt.close()



j = 2  # plane xD = 6.67
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
i = 0  # DX07
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX07_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX07_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 1  # DX10
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX10_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX10_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 2  # DX15
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_DX15_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_DX15_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i =  3  # ics, no jet
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],label_ics_low, label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
#plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],label_ics_high, label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks)
plt.grid()
plt.legend(loc='best',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG100_x10.pdf')
plt.show()
plt.close()

