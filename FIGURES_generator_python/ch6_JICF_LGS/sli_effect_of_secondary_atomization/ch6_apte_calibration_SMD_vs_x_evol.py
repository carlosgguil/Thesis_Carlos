
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

width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG


SMD_lim_along_z = (10,40)
ql_lim_along_z = (0,2.5)
z_lim = (0,33)
SMD_lim_along_y = (15,35)
ql_lim_along_y = (0,2.5)
y_lim = (-12.5,12.5)

# for maps
flux_levels_ = np.linspace(0,5.5,12)
SMD_levels_ = np.linspace(15,40,10)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/apte_model_calibration_u_vw_lognorm/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST/apte_model_calibration_u_vw_lognorm/'


folder_k1_0p05_k2_0p1 = folder + '/k1_0p05_k2_0p1/store_variables/'
folder_k1_0p05_k2_0p5 = folder + '/k1_0p05_k2_0p5/store_variables/'
folder_k1_0p05_k2_1p0 = folder + '/k1_0p05_k2_1p0/store_variables/'
folder_k1_0p10_k2_1p0 = folder + '/k1_0p10_k2_1p0/store_variables/'
folder_k1_0p20_k2_1p0 = folder + '/k1_0p20_k2_1p0/store_variables/'

formats = {'k1_0p05_k2_0p1':'-ok', 
           'k1_0p05_k2_0p5':'-^b', 
           'k1_0p05_k2_1p0':'-*r',
           'k1_0p10_k2_1p0':'--*g', 
           'k1_0p20_k2_1p0':'--*y'}

formats = {'k1_0p05_k2_0p1':'-b', 
           'k1_0p05_k2_0p5':'-r', 
           'k1_0p05_k2_1p0':'-k',
          'k1_0p10_k2_1p0':'--g', 
          'k1_0p20_k2_1p0':'--y'}


label_k1_0p05_k2_0p1 = r'$K_1 = 0.05, ~K_2 = 0.25$'
label_k1_0p05_k2_0p5 = r'$K_1 = 0.05, ~K_2 = 0.5$'
label_k1_0p05_k2_1p0 = r'$K_1 = 0.05, ~K_2 = 1.0$'
label_k1_0p10_k2_1p0 = r'$K_1 = 0.10, ~K_2 = 1.0$'
label_k1_0p20_k2_1p0 = r'$K_1 = 0.20, ~K_2 = 1.0$'

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





#%% expe data

# experimental SMD
SMD_expe = 31

# estimate expe errors
#error_SMD  = 0.26
#error_flux = 0.37
error_SMD  = 0.14
error_flux = 0.2


#%% get SMD evol data






# 

df_SMD_evol_k1_0p05_k2_0p1 = pd.read_csv(folder_k1_0p05_k2_0p1+'SMD_evolution.csv')
df_SMD_evol_k1_0p05_k2_0p5  = pd.read_csv(folder_k1_0p05_k2_0p5+'SMD_evolution.csv')
df_SMD_evol_k1_0p05_k2_1p0 = pd.read_csv(folder_k1_0p05_k2_1p0+'SMD_evolution.csv')
df_SMD_evol_k1_0p10_k2_1p0 = pd.read_csv(folder_k1_0p10_k2_1p0+'SMD_evolution.csv')
df_SMD_evol_k1_0p20_k2_1p0 = pd.read_csv(folder_k1_0p20_k2_1p0+'SMD_evolution.csv')

x_k1_0p05_k2_0p1 = df_SMD_evol_k1_0p05_k2_0p1['x'].values
x_k1_0p05_k2_0p1 = np.insert(x_k1_0p05_k2_0p1,0,5)
SMD_k1_0p05_k2_0p1 = df_SMD_evol_k1_0p05_k2_0p1[SMD_to_read].values
SMD_k1_0p05_k2_0p1 = np.insert(SMD_k1_0p05_k2_0p1,0,SMD_from_SPS)

x_k1_0p05_k2_0p5 = df_SMD_evol_k1_0p05_k2_0p5['x'].values
x_k1_0p05_k2_0p5 = np.insert(x_k1_0p05_k2_0p5,0,5)
SMD_k1_0p05_k2_0p5 = df_SMD_evol_k1_0p05_k2_0p5[SMD_to_read].values
SMD_k1_0p05_k2_0p5 = np.insert(SMD_k1_0p05_k2_0p5,0,SMD_from_SPS)

x_k1_0p05_k2_1p0 = df_SMD_evol_k1_0p05_k2_1p0['x'].values
x_k1_0p05_k2_1p0 = np.insert(x_k1_0p05_k2_1p0,0,5)
SMD_k1_0p05_k2_1p0 = df_SMD_evol_k1_0p05_k2_1p0[SMD_to_read].values
SMD_k1_0p05_k2_1p0 = np.insert(SMD_k1_0p05_k2_1p0,0,SMD_from_SPS)

x_k1_0p10_k2_1p0 = df_SMD_evol_k1_0p10_k2_1p0['x'].values
x_k1_0p10_k2_1p0 = np.insert(x_k1_0p10_k2_1p0,0,5)
SMD_k1_0p10_k2_1p0 = df_SMD_evol_k1_0p10_k2_1p0[SMD_to_read].values
SMD_k1_0p10_k2_1p0 = np.insert(SMD_k1_0p10_k2_1p0,0,SMD_from_SPS)

x_k1_0p20_k2_1p0 = df_SMD_evol_k1_0p20_k2_1p0['x'].values
x_k1_0p20_k2_1p0 = np.insert(x_k1_0p20_k2_1p0,0,5)
SMD_k1_0p20_k2_1p0 = df_SMD_evol_k1_0p20_k2_1p0[SMD_to_read].values
SMD_k1_0p20_k2_1p0 = np.insert(SMD_k1_0p20_k2_1p0,0,SMD_from_SPS)


# for plotting diameters
cases_all = ['k1_0p05_k2_0p1',
             'k1_0p05_k2_0p5',
             'k1_0p05_k2_1p0',
             'k1_0p10_k2_1p0',
             'k1_0p20_k2_1p0',]
SMDs_all = [SMD_k1_0p05_k2_0p1[-1], 
            SMD_k1_0p05_k2_0p5[-1],
            SMD_k1_0p05_k2_1p0[-1],
            SMD_k1_0p10_k2_1p0[-1],
            SMD_k1_0p20_k2_1p0[-1]]



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

ax1.plot(x_k1_0p05_k2_0p1,SMD_k1_0p05_k2_0p1, 
         formats['k1_0p05_k2_0p1'], label=label_k1_0p05_k2_0p1)
ax1.plot(x_k1_0p05_k2_0p5,SMD_k1_0p05_k2_0p5, 
         formats['k1_0p05_k2_0p5'], label=label_k1_0p05_k2_0p5)
ax1.plot(x_k1_0p05_k2_1p0,SMD_k1_0p05_k2_1p0, 
         formats['k1_0p05_k2_1p0'], label=label_k1_0p05_k2_1p0)
ax1.plot(x_k1_0p10_k2_1p0,SMD_k1_0p10_k2_1p0, 
         formats['k1_0p10_k2_1p0'], label=label_k1_0p10_k2_1p0)
ax1.plot(x_k1_0p20_k2_1p0,SMD_k1_0p20_k2_1p0, 
         formats['k1_0p20_k2_1p0'], label=label_k1_0p20_k2_1p0)
ax1.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)
# characteristics main plot
ax1.set_xlabel(label_x)
ax1.set_ylabel(label_SMD)
ax1.set_xlim(4,83)
ax1.set_ylim(00,85)
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
ip = InsetPosition(ax1, [0.55,0.40,0.3,0.5])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

# data for embedded plot
ax2.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax2.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
ax2.plot(x_k1_0p05_k2_0p1,SMD_k1_0p05_k2_0p1, 
         formats['k1_0p05_k2_0p1'])
ax2.plot(x_k1_0p05_k2_0p5,SMD_k1_0p05_k2_0p5, 
         formats['k1_0p05_k2_0p5'])
ax2.plot(x_k1_0p05_k2_1p0,SMD_k1_0p05_k2_1p0, 
         formats['k1_0p05_k2_1p0'])
ax2.plot(x_k1_0p10_k2_1p0,SMD_k1_0p10_k2_1p0, 
         formats['k1_0p10_k2_1p0'])
ax2.plot(x_k1_0p20_k2_1p0,SMD_k1_0p20_k2_1p0, 
         formats['k1_0p20_k2_1p0'])




# characteristics embedded plot
ax2.set_xlim((79,81))
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
plt.savefig(folder_manuscript+'SMD_vs_x_apte_calibration_comparison.pdf')
plt.show()
plt.close()


#%% Plot SMDs and deviations with experiments
for i in range(len(cases_all)):
    case_i = cases_all[i]
    SMD_i = SMDs_all[i]
    
    eps_SMD = (SMD_i - SMD_expe)/SMD_expe*100
    print(f'  Case {case_i}: SMD = {SMD_i:.3f}, error: {eps_SMD:.3f}')