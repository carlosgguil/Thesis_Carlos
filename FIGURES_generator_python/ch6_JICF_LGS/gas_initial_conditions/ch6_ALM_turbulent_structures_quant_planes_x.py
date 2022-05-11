# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:22:20 2021

@author: d601630
"""


#from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import fft
from scipy.fftpack import fftfreq
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)


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
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/gas_field_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/turbulence_state_u_mean_rms_probes/'
folder_ALM = folder + 'ALM_thesis/'


#%%  cases and labels

cases = [folder + 'SPS_UG100_DX10/',
         folder + '/noALM/',
         folder + 'ALM_flatBL_inclined_force_z_negative_original/',
         folder + 'custom_inlet_UG100_DX10_withRMS/']


cases_SPS = [folder + 'SPS_UG100_DX10/']
cases_ICS = [folder_ALM + 'no_ALM_no_droplets/',
             folder_ALM + 'ALM_initial_no_droplets/',
             folder_ALM + 'FDC_0p24_no_droplets/',
             folder_ALM + 'FDC_0p30_no_droplets/']
cases_LGS = [folder_ALM + 'no_ALM_with_droplets/',
             folder_ALM + 'ALM_initial_with_droplets/',
             folder_ALM + 'FDC_0p24_with_droplets/',
             folder_ALM + 'FDC_0p30_with_droplets/']


label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x ~[\mathrm{mm}]$'
label_z_ax   = '$z ~[\mathrm{mm}]$'

labels_cases_SPS = [r'$\mathrm{UG100\_DX10}$' ]
labels_cases_ICS = [r'$\mathrm{No~pert.}$',
                    r'$\mathrm{ALM~baseline}$' , 
                    r'$\mathrm{ALM~optimal}$', 
                    r'$\mathrm{ALM~modified}$']


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

formats_dat = ['k']
formats_ICS = ['b', 'r', 'y','g']
formats_LGS = ['--b', '--r', '--y','g']




'''
# test actuator forces
cases_ICS = [folder_ALM + 'no_ALM_no_droplets/',
             folder_ALM + 'ALM_initial_no_droplets/',
             folder_ALM + 'ALM_final_no_droplets/',
             folder_ALM + 'FDC_0p17/',
             folder_ALM + 'FDC_0p24/',
             folder_ALM + 'FDC_0p30/',
             folder_ALM + 'FDC_0p35/']
cases_LGS = cases_ICS
labels_cases_ICS = [r'$\mathrm{No~pert.}$',
                    r'$\mathrm{ALM~baseline}$' , 
                    r'$\mathrm{ALM~optimal}$',
                    r'F = 0.17', 
                    r'F = 0.24',
                    r'F = 0.30',
                    r'F = 0.35']
formats_ICS = ['b', 'r', 'g','--k','--b','--r','--y']
formats_LGS = formats_ICS
'''



y_coord_lim = (-8,8)
y_coord_ticks = [-8, -4, 0, 4, 8]
u_lim_UG75  = (30,85)
u_ticks_UG75 = [30, 40, 50, 60, 70, 80]
u_lim_UG100  = (20,120)
u_ticks_UG100 = [20, 40, 60, 80, 100, 120]
u_lim_UG75  = u_lim_UG100
u_ticks_UG75 = u_ticks_UG100

#y_ticks_u_vs_x = [-20, 0, 20, 40, 60, 80, 100, 120]

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_y_ax   = '$y ~[\mathrm{mm}]$'

# For z locations
k_low = 4; k_high = -1 
#k_low = 2; k_high = -3

#%% get data

# Define arrays
y_values      = [ [] for m in range(len(cases)) ]
u_mean_values = [ [] for m in range(len(cases)) ]
v_mean_values = [ [] for m in range(len(cases)) ]
w_mean_values = [ [] for m in range(len(cases)) ]

for i in range(len(cases_SPS)):
    
    # x planes
    case_i_x02p5 = cases[i]+'probes_turb_gas_plane_x02p5/'
    case_i_x05   = cases[i]+'probes_turb_gas_plane_x05/'
    case_i_x10   = cases[i]+'probes_turb_gas_plane_x10/'
    case_i_x15   = cases[i]+'probes_turb_gas_plane_x15/'
    
    if os.path.exists(case_i_x02p5):
        IS_CUSTOM_INLET = False
    else:
        IS_CUSTOM_INLET = True
        
    
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
            if IS_CUSTOM_INLET and j==0:
                y_values[i][j][k] = np.nan
                u_mean_values[i][j][k] = np.nan
                v_mean_values[i][j][k] = np.nan
                w_mean_values[i][j][k] = np.nan
            else:
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


#%% get ICS and LGS data


lines_at_x05 = ['planex05_z01p6.csv', 'planex05_z05p0.csv']
lines_at_x10 = ['planex10_z01p6.csv', 'planex10_z05p0.csv']

y_values_x05_lines_ICS = [[] for i in range(len(cases_ICS))]
y_values_x05_lines_LGS = [[] for i in range(len(cases_ICS))]
u_values_x05_lines_ICS = [[] for i in range(len(cases_ICS))]
u_values_x05_lines_LGS = [[] for i in range(len(cases_ICS))]
y_values_x10_lines_ICS = [[] for i in range(len(cases_ICS))]
y_values_x10_lines_LGS = [[] for i in range(len(cases_ICS))]
u_values_x10_lines_ICS = [[] for i in range(len(cases_ICS))]
u_values_x10_lines_LGS = [[] for i in range(len(cases_ICS))]
for i in range(len(cases_ICS)):
    case_ICS_i = cases_ICS[i]
    case_LGS_i = cases_LGS[i]
    for j in range(len(lines_at_x05)):
        df_ICS = pd.read_csv(case_ICS_i+lines_at_x05[j])
        y_values_x05_lines_ICS[i].append(df_ICS['Points_1'].values*1e3)
        u_values_x05_lines_ICS[i].append(df_ICS['U_MEAN_0'].values)
        
        df_LGS = pd.read_csv(case_LGS_i+lines_at_x05[j])
        y_values_x05_lines_LGS[i].append(df_LGS['Points_1'].values*1e3)
        u_values_x05_lines_LGS[i].append(df_LGS['U_MEAN_0'].values)

    for j in range(len(lines_at_x10)):
        df_ICS = pd.read_csv(case_ICS_i+lines_at_x10[j])
        y_values_x10_lines_ICS[i].append(df_ICS['Points_1'].values*1e3)
        u_values_x10_lines_ICS[i].append(df_ICS['U_MEAN_0'].values)
        
        df_LGS = pd.read_csv(case_LGS_i+lines_at_x10[j])
        y_values_x10_lines_LGS[i].append(df_LGS['Points_1'].values*1e3)
        u_values_x10_lines_LGS[i].append(df_LGS['U_MEAN_0'].values) 

u_to_plot = u_mean_values


#%% line x = 5 mm, z = 1.6 mm

#SPS
j = 1 # x = 5 mm
k = 3 # z = 1.6 mm
plt.figure(figsize=figsize_)
plt.title(labels_z_planes[k])
i = 0; plt.plot(y_values[i][j][k],u_to_plot[i][j][k],'k', label=labels_cases_SPS[i])

# ICS/LGS
k = 0
for i in range(len(cases_ICS)):
    plt.plot(y_values_x05_lines_ICS[i][k],u_values_x05_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(y_values_x05_lines_LGS[i][k],u_values_x05_lines_LGS[i][k],
             formats_LGS[i])
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG100)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_line_x05_z01p6_ux_mean_along_y.pdf')
plt.show()
plt.close()



#%% line x = 5 mm, z = 5 mm

#SPS
j = 1 # x = 5 mm
k = -2 # z = 5 mm
plt.figure(figsize=figsize_)
plt.title(labels_z_planes[k])
i = 0; plt.plot(y_values[i][j][k],u_to_plot[i][j][k],'k', label=labels_cases_SPS[i])

# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    plt.plot(y_values_x05_lines_ICS[i][k],u_values_x05_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(y_values_x05_lines_LGS[i][k],u_values_x05_lines_LGS[i][k],
             formats_LGS[i])
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG100)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
#plt.savefig(folder_manuscript+'ALM_line_x05_z05p0_ux_mean_along_y.pdf')
plt.show()
plt.close()


#%% line x = 5 mm, z = 5 mm with zoom


j = 1 # x = 5 mm
k = -2 # z = 5 mm
fig, ax1 = plt.subplots(figsize=figsize_)
plt.title(labels_z_planes[k])

# data for main plot
# SPS
i = 0;  ax1.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i], label=labels_cases_SPS[i])

#custom (filtered)
i = 1
#plt.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i], label=labels_cases_dat[i])



# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    ax1.plot(y_values_x05_lines_ICS[i][k],u_values_x05_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax1.plot(y_values_x05_lines_LGS[i][k],u_values_x05_lines_LGS[i][k],
             formats_LGS[i])
ax1.set_xlabel(label_y_ax)
ax1.set_ylabel(label_u_ax)
ax1.set_xlim(y_coord_lim)
ax1.set_ylim(u_lim_UG100)
ax1.set_xticks(y_coord_ticks)
ax1.set_yticks(u_ticks_UG100)
ax1.grid()
#plt.legend(loc='best')


# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.25,0.25,0.5,0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec='0.5')

# data for embedded plot
j = 1 # x = 5 mm
k = -2 # z = 5 mm
# SPS
i = 0; ax2.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i], label=labels_cases_SPS[i])

# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    ax2.plot(y_values_x05_lines_ICS[i][k],u_values_x05_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax2.plot(y_values_x05_lines_LGS[i][k],u_values_x05_lines_LGS[i][k],
             formats_LGS[i])





# characteristics embedded plot
ax2.set_xlim((-2,2))
ax2.set_ylim((98,112))
#ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 60*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')

# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_line_x05_z05p0_ux_mean_along_y.pdf')
plt.show()
plt.close()




#%% line x = 10 mm, z = 1.6 mm

#SPS
j = 2 # x = 10 mm
k = 3 # z = 1.6 mm
plt.figure(figsize=figsize_)
plt.title(labels_z_planes[k])
i = 0; plt.plot(y_values[i][j][k],u_to_plot[i][j][k],'k', label=f'{labels_cases_SPS[i]}')

# ICS/LGS
k = 0
for i in range(len(cases_ICS)):
    plt.plot(y_values_x10_lines_ICS[i][k],u_values_x10_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(y_values_x10_lines_LGS[i][k],u_values_x10_lines_LGS[i][k],
             formats_LGS[i])
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
plt.ylim(u_lim_UG100)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_line_x10_z01p6_ux_mean_along_y.pdf')
plt.show()
plt.close()



#%% line x = 10 mm, z = 5 mm

#SPS
j = 2 # x = 10 mm
k = -2 # z = 5 mm
plt.figure(figsize=figsize_)
plt.title(labels_z_planes[k])
i = 0; plt.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i], label=labels_cases_SPS[i])

# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    plt.plot(y_values_x10_lines_ICS[i][k],u_values_x10_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(y_values_x10_lines_LGS[i][k],u_values_x10_lines_LGS[i][k],
             formats_LGS[i])
plt.xlabel(label_y_ax)
plt.ylabel(label_u_ax)
plt.xlim(y_coord_lim)
#plt.ylim(u_lim_UG100)
plt.ylim(20,120)
plt.xticks(y_coord_ticks)
plt.yticks(u_ticks_UG100)
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_line_x10_z05p0_ux_mean_along_y.pdf')
plt.show()
plt.close()



#%% line x = 10 mm, z = 5 mm con zoom


j = 2 # x = 10 mm
k = -2 # z = 5 mm
fig, ax1 = plt.subplots(figsize=figsize_)
plt.title(labels_z_planes[k])

# data for main plot
# SPS
i = 0; ax1.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i],  label=labels_cases_SPS[i])



# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    ax1.plot(y_values_x10_lines_ICS[i][k],u_values_x10_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax1.plot(y_values_x10_lines_LGS[i][k],u_values_x10_lines_LGS[i][k],
             formats_LGS[i])
ax1.set_xlabel(label_y_ax)
ax1.set_ylabel(label_u_ax)
ax1.set_xlim(y_coord_lim)
ax1.set_ylim(u_lim_UG100)
ax1.set_xticks(y_coord_ticks)
ax1.set_yticks(u_ticks_UG100)
ax1.grid()
#plt.legend(loc='best')


# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.25,0.09,0.5,0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec='0.5')

# data for embedded plot
j = 2 # x = 10 mm
k = -2 # z = 5 mm
# SPS
i = 0; ax2.plot(y_values[i][j][k],u_to_plot[i][j][k],formats_dat[i], label=labels_cases_SPS[i])

# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    ax2.plot(y_values_x10_lines_ICS[i][k],u_values_x10_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax2.plot(y_values_x10_lines_LGS[i][k],u_values_x10_lines_LGS[i][k],
             formats_LGS[i])





# characteristics embedded plot
ax2.set_xlim((-2,2))
ax2.set_ylim((72,112))
#ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 60*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')

# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_line_x10_z05p0_ux_mean_along_y.pdf')
plt.show()
plt.close()