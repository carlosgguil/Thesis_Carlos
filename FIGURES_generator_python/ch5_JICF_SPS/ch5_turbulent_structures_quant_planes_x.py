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
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/turbulent_structures/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/turbulence_state_u_mean_rms_probes/'



#%%  cases and labels

cases = [folder + 'SPS_UG75_DX10/',
         folder + 'SPS_UG75_DX20/',
         folder + 'ics_unperturbed_2nd_op/',
         folder + 'SPS_UG100_DX10/',
         folder + 'SPS_UG100_DX20/',
         folder + 'ics_unperturbed_1st_op/']


label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x ~[\mathrm{mm}]$'
label_z_ax   = '$z ~[\mathrm{mm}]$'


labels_cases = [r'$\mathrm{UG75}\_\mathrm{DX10}$' ,r'$\mathrm{UG75}\_\mathrm{DX20}$', r'$\mathrm{UG75~no~jet}$',
                r'$\mathrm{UG100}\_\mathrm{DX10}$' , r'$\mathrm{UG100}\_\mathrm{DX20}$', r'$\mathrm{UG100~no~jet}$']



formats = ['k','--k',':k','b','--b',':b']


labels_x_planes = [r'$x = 2.5~\mathrm{mm}$',  
                   r'$x = 5~\mathrm{mm}$', 
                   r'$x = 10~\mathrm{mm}$',
                   r'$x = 15~\mathrm{mm}$', 
                   r'$x = 20~\mathrm{mm}$']

labels_z_planes = [r'$z = 0.2~\mathrm{mm}$',  
                   r'$z = 0.8~\mathrm{mm}$', 
                   r'$z = 1~\mathrm{mm}$',  
                   r'$z = 1.6~\mathrm{mm}$', 
                   r'$z = 2~\mathrm{mm}$', 
                   r'$z = 3~\mathrm{mm}$', 
                   r'$z = 4~\mathrm{mm}$', 
                   r'$z = 5~\mathrm{mm}$', 
                   r'$z = 6,\mathrm{mm}$'] 


y_coord_lim = (-8,8)
y_coord_ticks = [-8, -4, 0, 4, 8]
u_lim_UG75  = (30,85)
u_ticks_UG75 = [0,30, 40, 50, 60, 70, 80]
u_lim_UG100  = (0,120)#(30,120)
u_ticks_UG100 = [0,30, 60, 90, 120]
u_lim_UG75  = u_lim_UG100
u_ticks_UG75 = u_ticks_UG100

#y_ticks_u_vs_x = [-20, 0, 20, 40, 60, 80, 100, 120]

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_y_ax   = '$y ~[\mathrm{mm}]$'

# For z locations
k_low = 3; k_high = -2
#k_low = 2; k_high = -3

#%% get data

# Define arrays
y_values      = [ [] for m in range(len(cases)) ]
u_mean_values = [ [] for m in range(len(cases)) ]
v_mean_values = [ [] for m in range(len(cases)) ]
w_mean_values = [ [] for m in range(len(cases)) ]

for i in range(len(cases)):
    
    # x planes
    case_i_x02p5 = cases[i]+'probes_turb_gas_plane_x02p5/'
    case_i_x05   = cases[i]+'probes_turb_gas_plane_x05/'
    case_i_x10   = cases[i]+'probes_turb_gas_plane_x10/'
    case_i_x15   = cases[i]+'probes_turb_gas_plane_x15/'
    
    # put all folders together
    cases_x_planes = [case_i_x02p5, 
                      case_i_x05, 
                      case_i_x10, 
                      case_i_x15]
    names_planes = ['planeX02p5', 'planeX05', 'planeX10', 'planeX15']
    
    # add arrays per each plane
    y_values[i]      = [ [] for m in range(len(cases_x_planes)) ]
    u_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    v_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    w_mean_values[i] = [ [] for m in range(len(cases_x_planes)) ]
    for j in range(len(cases_x_planes)):
        try:
            # z lines
            line_z00p2 = cases_x_planes[j]+'line_'+names_planes[j]+'_z00p2mm_U_MEAN.dat'
            line_z00p8 = cases_x_planes[j]+'line_'+names_planes[j]+'_z00p8mm_U_MEAN.dat'
            line_z01   = cases_x_planes[j]+'line_'+names_planes[j]+'_z01mm_U_MEAN.dat'
            line_z01p6 = cases_x_planes[j]+'line_'+names_planes[j]+'_z01p6mm_U_MEAN.dat'
            line_z02   = cases_x_planes[j]+'line_'+names_planes[j]+'_z02mm_U_MEAN.dat'
            line_z03   = cases_x_planes[j]+'line_'+names_planes[j]+'_z03mm_U_MEAN.dat'
            line_z03p2 = cases_x_planes[j]+'line_'+names_planes[j]+'_z03p2mm_U_MEAN.dat'
            line_z04   = cases_x_planes[j]+'line_'+names_planes[j]+'_z04mm_U_MEAN.dat'
            line_z05   = cases_x_planes[j]+'line_'+names_planes[j]+'_z05mm_U_MEAN.dat'
            line_z06   = cases_x_planes[j]+'line_'+names_planes[j]+'_z06mm_U_MEAN.dat'
            
            lines = [line_z00p2, line_z00p8, line_z01, line_z01p6, line_z02,
                     line_z03, line_z03p2, line_z04, line_z05, line_z06]    
            
            # add arrays per each plane
            y_values[i][j]      = [ [] for m in range(len(lines)) ]
            u_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            v_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            w_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            
            for k in range(len(lines)):
                probe = pd.read_csv(lines[k],sep='(?<!\\#)\s+',engine='python')
                probe.columns =  [name.split(':')[1] for name in probe.columns] 
                
                # Get indices where time instants changes
                indices = [-1]
                for n in range(len(probe)):
                    if np.isnan(probe.loc[n]['total_time']):
                        indices.append(n)
                        
                # recover mean values from last time instant
                df = probe.loc[indices[-2]+1:indices[-1]-1]
                
                
                y_values[i][j][k]      = df['Y'].values*1e3
                u_mean_values[i][j][k] = df['U_MEAN(1)'].values
                v_mean_values[i][j][k] = df['U_MEAN(2)'].values
                w_mean_values[i][j][k] = df['U_MEAN(3)'].values
        except:
            # z lines
            line_z00p2 = cases_x_planes[j]+'z00p2mm.csv'
            line_z00p8 = cases_x_planes[j]+'z00p8mm.csv'
            line_z01   = cases_x_planes[j]+'z01mm.csv'
            line_z01p6 = cases_x_planes[j]+'z01p6mm.csv'
            line_z02   = cases_x_planes[j]+'z02mm.csv'
            line_z03   = cases_x_planes[j]+'z03mm.csv'
            line_z03p2 = cases_x_planes[j]+'z03p2mm.csv'
            line_z04   = cases_x_planes[j]+'z04mm.csv'
            line_z05   = cases_x_planes[j]+'z05mm.csv'
            line_z06   = cases_x_planes[j]+'z06mm.csv'
            
            lines = [line_z00p2, line_z00p8, line_z01, line_z01p6, line_z02,
                     line_z03, line_z03p2, line_z04, line_z05, line_z06]    
            
            # add arrays per each plane
            y_values[i][j]      = [ [] for m in range(len(lines)) ]
            u_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            v_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            w_mean_values[i][j] = [ [] for m in range(len(lines)) ]
            
            for k in range(len(lines)):
                try:
                    df = pd.read_csv(lines[k])
                    y_values[i][j][k]      = df['Points_1'].values*1e3
                    u_mean_values[i][j][k] = df['U_MEAN_0'].values
                    v_mean_values[i][j][k] = df['U_MEAN_1'].values
                    w_mean_values[i][j][k] = df['U_MEAN_2'].values
                except:
                    y_values[i][j][k]      = None
                    u_mean_values[i][j][k] = None
                    v_mean_values[i][j][k] = None
                    w_mean_values[i][j][k] = None



#%% Plots UG75

u_to_plot = u_mean_values




j = 1  # plane x = 5 mm
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
i = 0  # case UG75, DX10
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 1  # case UG75, DX20
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'--k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'--b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 2  # case No jet
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],':k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],':b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG75)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG75)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG75_x05.pdf')
plt.show()
plt.close()


j = 2  # plane x = 10 mm
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
# k_low
i = 0; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
i = 1; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'--k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
i = 2; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],':k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
# k_high
i = 0; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 1; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'--b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 2; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],':b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG75)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG75)
plt.grid()
plt.legend(loc='lower left',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG75_x10.pdf')
plt.show()
plt.close()

#%% Plots UG100





j = 1  # plane x = 5 mm
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
i = 3  # case UG100, DX10
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 4  # case UG100, DX20
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'--k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'--b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 5  # case UG100, no jet
plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],':k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],':b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG100)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG100_x05.pdf')
plt.show()
plt.close()


j = 2  # plane x = 10 mm
plt.figure(figsize=figsize_)
plt.title(labels_x_planes[j])
# k_low
i = 3; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
i = 4; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],'--k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
i = 5; plt.plot(y_values[i][j][k_low],u_to_plot[i][j][k_low],':k', label=f'{labels_cases[i]}, {labels_z_planes[k_low]}')
# k_high
i = 3; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 4; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],'--b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
i = 5; plt.plot(y_values[i][j][k_high],u_to_plot[i][j][k_high],':b', label=f'{labels_cases[i]}, {labels_z_planes[k_high]}')
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG100)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
plt.legend(loc='lower left',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_iso-x_along_y_ux_mean_UG100_x10.pdf')
plt.show()
plt.close()

