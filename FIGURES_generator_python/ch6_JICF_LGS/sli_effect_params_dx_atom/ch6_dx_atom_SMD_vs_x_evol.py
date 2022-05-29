
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""





import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import pickle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import numpy as np
import pandas as pd
from functions_expe_comparison import average_along_y, average_along_z
from functions_expe_comparison import get_SMD_from_integrated_profile, get_SMD_flux_weighted
from sprPost_calculations import get_discrete_spray, get_sprays_list
from sprPost_functions import get_grids_common_boundaries
import sprPost_plot as sprPlot


# Change size of figures 
FFIG = 0.5
#mpl.rcParams['font.size'] = 40*fPic
plt.rcParams['xtick.labelsize']  = 50*FFIG
plt.rcParams['ytick.labelsize']  = 50*FFIG
plt.rcParams['axes.labelsize']   = 50*FFIG
plt.rcParams['axes.labelpad']    = 30*FFIG
plt.rcParams['axes.titlesize']   = 50*FFIG
plt.rcParams['legend.fontsize']  = 40*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 40*FFIG #20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*18,FFIG*13)
figsize_expe = (FFIG*21.7,FFIG*12.4)
figsize_along_z  = (FFIG*10,FFIG*15)

label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'




SMD_lim_along_z = (10,40)
ql_lim_along_z = (0,2.5)
z_lim = (0,33)
SMD_lim_along_y = (15,35)
ql_lim_along_y = (0,2.5)
y_lim = (-12.5,12.5)

# for maps
flux_levels_ = np.linspace(0,5.5,12)
SMD_levels_ = np.linspace(15,40,10)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_dx_atom/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'



folder_dx00 = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_dx04 = folder + '/param_dx_atom/dx_atom_04/store_variables/'
folder_dx08 = folder + '/param_dx_atom/dx_atom_08/store_variables/'
folder_dx12 = folder + '/param_dx_atom/dx_atom_12/store_variables/'
folder_dx16 = folder + '/param_dx_atom/dx_atom_16/store_variables/'
folder_dx20 = folder + '/param_dx_atom/dx_atom_20/store_variables/'


formats = {'dx00':'-ok', 'dx04':'-^b', 'dx08':'-*r', 
           'dx12':'-^g', 'dx16':'-oy', 'dx20':'-^k'}

formats = {'dx00':'-k', 'dx04':'-b', 'dx08':'-r', 
           'dx12':'-g', 'dx16':'-y', 'dx20':'--k'}



SMD_to_read = 'SMD_FW' # 'SMD', 'SMD_FW'

SMD_from_SPS = 80.2

#%% Experimental data and simulation parameters (do not touch)
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'
data_int_y_exp = pd.read_csv(folder_expe + '1210_01_data_integrated_y_exp.csv')
data_int_z_exp = pd.read_csv(folder_expe + '1210_01_data_integrated_z_exp.csv')

z_int_exp  = data_int_y_exp['z_values']
flux_z_exp = data_int_y_exp['flux_z_exp']
SMD_z_exp  = data_int_y_exp['SMD_z_exp']

y_int_exp  = data_int_z_exp['y_values']
flux_y_exp = data_int_z_exp['flux_y_exp']
SMD_y_exp  = data_int_z_exp['SMD_y_exp']

# estimate errors
error_SMD  = 0.14
error_flux = 0.2

error_q_y_expe = flux_y_exp*error_flux
error_q_z_expe = flux_z_exp*error_flux
error_SMD_y_expe = SMD_y_exp*error_SMD
error_SMD_z_expe = SMD_z_exp*error_SMD


params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']


# expe data for maps
with open(folder_expe+'/pickle_map_expe_high_we', 'rb') as f:
    obj = pickle.load(f)
    expe_y_values = obj[0]
    expe_z_values = obj[1]
    expe_yy_values = obj[2]
    expe_zz_values = obj[3]
    expe_flux_values = obj[4]
    expe_SMD_values = obj[5]



width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG


#%% expe data

# experimental SMD
SMD_expe = 31

# estimate expe errors
#error_SMD  = 0.26
#error_flux = 0.37
error_SMD  = 0.14
error_flux = 0.2


#%% get SMD evol 

label_dx00 = r'$\Delta x_\mathrm{atom} = 0~\mathrm{mm}$'
label_dx04 = r'$\Delta x_\mathrm{atom} = 4~\mathrm{mm}$'
label_dx08 = r'$\Delta x_\mathrm{atom} = 8~\mathrm{mm}$'
label_dx12 = r'$\Delta x_\mathrm{atom} = 12~\mathrm{mm}$'
label_dx16 = r'$\Delta x_\mathrm{atom} = 16~\mathrm{mm}$'
label_dx20 = r'$\Delta x_\mathrm{atom} = 20~\mathrm{mm}$'


x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)
# 

df_SMD_evol_dx_00 = pd.read_csv(folder_dx00+'SMD_evolution.csv')
df_SMD_evol_dx_04 = pd.read_csv(folder_dx04+'SMD_evolution.csv')
df_SMD_evol_dx_08 = pd.read_csv(folder_dx08+'SMD_evolution.csv')
df_SMD_evol_dx_12 = pd.read_csv(folder_dx12+'SMD_evolution.csv')
df_SMD_evol_dx_16 = pd.read_csv(folder_dx16+'SMD_evolution.csv')
df_SMD_evol_dx_20 = pd.read_csv(folder_dx20+'SMD_evolution.csv')

x_dx_00 = df_SMD_evol_dx_00['x'].values
x_dx_00 = np.insert(x_dx_00,0,5)
SMD_dx_00 = df_SMD_evol_dx_00[SMD_to_read].values
SMD_dx_00 = np.insert(SMD_dx_00,0,SMD_from_SPS)

x_dx_04 = df_SMD_evol_dx_04['x'].values
x_dx_04 = np.insert(x_dx_04,0,5)
SMD_dx_04 = df_SMD_evol_dx_04[SMD_to_read].values
SMD_dx_04 = np.insert(SMD_dx_04,0,SMD_from_SPS)
SMD_dx_04[:5] = SMD_from_SPS

x_dx_08 = df_SMD_evol_dx_08['x'].values
x_dx_08 = np.insert(x_dx_08,0,5)
SMD_dx_08 = df_SMD_evol_dx_08[SMD_to_read].values
SMD_dx_08 = np.insert(SMD_dx_08,0,SMD_from_SPS)
SMD_dx_08[:6] = SMD_from_SPS

x_dx_12 = df_SMD_evol_dx_12['x'].values
x_dx_12 = np.insert(x_dx_12,0,5)
SMD_dx_12 = df_SMD_evol_dx_12[SMD_to_read].values
SMD_dx_12 = np.insert(SMD_dx_12,0,SMD_from_SPS)
SMD_dx_12[:8] = SMD_from_SPS

x_dx_16 = df_SMD_evol_dx_16['x'].values
x_dx_16 = np.insert(x_dx_16,0,5)
SMD_dx_16 = df_SMD_evol_dx_16[SMD_to_read].values
SMD_dx_16 = np.insert(SMD_dx_16,0,SMD_from_SPS)
SMD_dx_16[:11] = SMD_from_SPS

x_dx_20 = df_SMD_evol_dx_20['x'].values
x_dx_20 = np.insert(x_dx_20,0,5)
SMD_dx_20 = df_SMD_evol_dx_20[SMD_to_read].values
SMD_dx_20 = np.insert(SMD_dx_20,0,SMD_from_SPS)
SMD_dx_20[:11] = SMD_from_SPS



#%% single plot


plt.figure(figsize=figsize_)
plt.plot(x_dx_00,SMD_dx_00, formats['dx00'], label=label_dx00)
plt.plot(x_dx_04,SMD_dx_04, formats['dx04'], label=label_dx04)
plt.plot(x_dx_08,SMD_dx_08, formats['dx08'], label=label_dx08)
plt.plot(x_dx_12,SMD_dx_12, formats['dx12'], label=label_dx12)
plt.plot(x_dx_16,SMD_dx_16, formats['dx16'], label=label_dx16)
plt.plot(x_dx_20,SMD_dx_20, formats['dx20'], label=label_dx20)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_x)
plt.ylabel(label_SMD) 
plt.legend(loc='best', ncol=2)
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks(x_ticks_SMD_evol)
plt.yticks(y_ticks_SMD_evol)
plt.xlim(4.8,80.2)#(plot_bounds[0])
plt.ylim(00,80)#(plot_bounds[1])
#plt.savefig(folder_manuscript+'SMD_vs_x_dx_atom_comparison.pdf')
#plt.show()
plt.close()      


#%%

figsize_SMD_evol_along_x = (FFIG*28,FFIG*13)

x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)

'''
label_dx16_ETAB = r'$16~\mathrm{mm},~ETAB$'
df_SMD_evol_dx_16_ETAB = pd.read_csv(folder_dx16_ETAB+'SMD_evolution.csv')

x_dx_16_ETAB = df_SMD_evol_dx_16_ETAB['x'].values
x_dx_16_ETAB = np.insert(x_dx_16_ETAB,0,5)
SMD_dx_16_ETAB = df_SMD_evol_dx_16_ETAB[SMD_to_read].values
SMD_dx_16_ETAB = np.insert(SMD_dx_16_ETAB,0,SMD_from_SPS)
'''

fig, ax1 = plt.subplots(figsize=figsize_SMD_evol_along_x)

# data for main plot
# Experimental result
ax1.plot(x_dx_00,SMD_dx_00, formats['dx00'], label=label_dx00)
ax1.plot(x_dx_04,SMD_dx_04, formats['dx04'], label=label_dx04)
ax1.plot(x_dx_08,SMD_dx_08, formats['dx08'], label=label_dx08)
ax1.plot(x_dx_12,SMD_dx_12, formats['dx12'], label=label_dx12)
ax1.plot(x_dx_16,SMD_dx_16, formats['dx16'], label=label_dx16)
ax1.plot(x_dx_20,SMD_dx_20, formats['dx20'], label=label_dx20)
ax1.scatter(80,SMD_expe,color='black',marker='s',label=label_expe)
ax1.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)

# characteristics main plot
ax1.set_xlabel(label_x)
ax1.set_ylabel(label_SMD)
ax1.set_xlim(4.8,83)
ax1.set_ylim(00,82)
ax1.set_xticks(x_ticks_SMD_evol)
ax1.set_yticks(y_ticks_SMD_evol)
#ax1.legend(loc='best',ncol=2)
ax1.legend(bbox_to_anchor=(1.40,0.75), ncol=1)
ax1.grid()
#ax1.grid(which='major',linestyle='-',linewidth=4*FFIG)
#ax1.grid(which='minor',linestyle='--')

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.55,0.55,0.4,0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

# data for embedded plot
ax2.plot(x_dx_00,SMD_dx_00, formats['dx00'], label=label_dx00)
ax2.plot(x_dx_04,SMD_dx_04, formats['dx04'], label=label_dx04)
ax2.plot(x_dx_08,SMD_dx_08, formats['dx08'], label=label_dx08)
ax2.plot(x_dx_12,SMD_dx_12, formats['dx12'], label=label_dx12)
ax2.plot(x_dx_16,SMD_dx_16, formats['dx16'], label=label_dx16)
ax2.plot(x_dx_20,SMD_dx_20, formats['dx20'], label=label_dx20)


ax2.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax2.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)

# characteristics embedded plot
ax2.set_xlim((78,82))
ax2.set_ylim((15,36))
#ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 40*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')


# draw rectangle
'''
w_rect = ax2.get_xlim()[1] - ax2.get_xlim()[0]+0.6
h_rect = ax2.get_ylim()[1] - ax2.get_ylim()[0]
rect = Rectangle((ax2.get_xlim()[0]-0.3,ax2.get_ylim()[0]),w_rect,h_rect, 
                 linewidth=1,edgecolor='k',facecolor='none',zorder = 2)
ax1.add_patch(rect)
'''

# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_vs_x_dx_atom_comparison.pdf')
plt.show()
plt.close()
