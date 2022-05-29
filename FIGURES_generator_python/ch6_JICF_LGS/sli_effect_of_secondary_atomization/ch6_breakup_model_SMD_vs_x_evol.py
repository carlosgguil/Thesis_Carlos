
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
plt.rcParams['legend.fontsize']  = 35*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 30*FFIG #20*FFIG
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

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_breakup_model/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'



folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_TAB  = folder + '/param_breakup_model/TAB_u_vw_LN/store_variables/'
folder_ETAB = folder + '/param_breakup_model/ETAB_u_vw_LN/store_variables/'

formats = {'APTE':'-ok', 'TAB':'-^b', 'ETAB':'-*r'}
formats = {'APTE':'-k', 'TAB':'-b', 'ETAB':'-r'}

SMD_to_read = 'SMD_FW' # 'SMD', 'SMD_FW'


#SPS
format_SPS = '--^k'
label_SPS = r'$\mathrm{UG}100\_\mathrm{DX}10$'
SMD_from_SPS = 80.2

x_SPS     = [5, 10]
SMD_SPS_x = [SMD_from_SPS, 79.9]

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


#%% get SMD evol and plot

label_APTE = r'$\mathrm{Gorokhovski}$'
label_TAB = r'$\mathrm{TAB}$'
label_ETAB = r'$\mathrm{ETAB}$'

x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)
# 

df_SMD_evol_APTE = pd.read_csv(folder_APTE+'SMD_evolution.csv')
df_SMD_evol_TAB  = pd.read_csv(folder_TAB+'SMD_evolution.csv')
df_SMD_evol_ETAB = pd.read_csv(folder_ETAB+'SMD_evolution.csv')

x_APTE = df_SMD_evol_APTE['x'].values
x_APTE = np.insert(x_APTE,0,5)
SMD_APTE = df_SMD_evol_APTE[SMD_to_read].values
SMD_APTE = np.insert(SMD_APTE,0,SMD_from_SPS)

x_TAB = df_SMD_evol_TAB['x'].values
x_TAB = np.insert(x_TAB,0,5)
SMD_TAB = df_SMD_evol_TAB[SMD_to_read].values
SMD_TAB = np.insert(SMD_TAB,0,SMD_from_SPS)

x_ETAB = df_SMD_evol_ETAB['x'].values
x_ETAB = np.insert(x_ETAB,0,5)
SMD_ETAB = df_SMD_evol_ETAB[SMD_to_read].values
SMD_ETAB = np.insert(SMD_ETAB,0,SMD_from_SPS)


# for plotting diameters
cases_all = ['goro',
             'TAB',
             'ETAB']
             
SMDs_all = [SMD_APTE[-1], 
            SMD_TAB[-1],
            SMD_ETAB[-1]]

#%% 

plt.figure(figsize=figsize_)
plt.plot(x_APTE,SMD_APTE, formats['APTE'], label=label_APTE)
plt.plot(x_TAB,SMD_TAB, formats['TAB'], label=label_TAB)
plt.plot(x_ETAB,SMD_ETAB, formats['ETAB'], label=label_ETAB)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_x) 
plt.ylabel(label_SMD) 
plt.legend(loc='best', ncol=1)
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks(x_ticks_SMD_evol)
plt.yticks(y_ticks_SMD_evol)
plt.xlim(4.8,80.2)#(plot_bounds[0])
plt.ylim(00,80)#(plot_bounds[1])
#plt.savefig(folder_manuscript+'SMD_vs_x_breakup_models_comparison.pdf')
plt.show()
plt.close()      

#%% plot

figsize_SMD_evol_along_x = (FFIG*30,FFIG*12)


x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)


fig, ax1 = plt.subplots(figsize=figsize_SMD_evol_along_x)

# data for main plot
# Experimentql result
ax1.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax1.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)

ax1.plot(x_APTE,SMD_APTE, formats['APTE'], label=label_APTE)
ax1.plot(x_TAB,SMD_TAB, formats['TAB'], label=label_TAB)
ax1.plot(x_ETAB,SMD_ETAB, formats['ETAB'], label=label_ETAB)
ax1.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)
# characteristics main plot
ax1.set_xlabel(label_x)
ax1.set_ylabel(label_SMD)
ax1.set_xlim(4,83)
ax1.set_ylim(00,85)
ax1.set_xticks(x_ticks_SMD_evol)
ax1.set_yticks(y_ticks_SMD_evol)
#ax1.legend(loc='best',ncol=2)
ax1.legend(bbox_to_anchor=(1.25,0.75), ncol=1)
ax1.grid()
#ax1.grid(which='major',linestyle='-',linewidth=4*FFIG)
#ax1.grid(which='minor',linestyle='--')

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.55,0.40,0.3,0.5])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

# data for embedded plot
ax2.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax2.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
ax2.plot(x_APTE,SMD_APTE, formats['APTE'], label=label_APTE)
ax2.plot(x_TAB,SMD_TAB, formats['TAB'], label=label_TAB)
ax2.plot(x_ETAB,SMD_ETAB, formats['ETAB'], label=label_ETAB)
ax2.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)




# characteristics embedded plot
ax2.set_xlim((79,81))
ax2.set_ylim((12,36))
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
plt.savefig(folder_manuscript+'SMD_vs_x_breakup_models_comparison.pdf')
plt.show()
plt.close()


#%% Plot SMDs and deviations with experiments
for i in range(len(cases_all)):
    case_i = cases_all[i]
    SMD_i = SMDs_all[i]
    
    eps_SMD = (SMD_i - SMD_expe)/SMD_expe*100
    print(f'  Case {case_i}: SMD = {SMD_i:.3f}, error: {eps_SMD:.3f}')