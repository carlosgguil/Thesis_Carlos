
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

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'



#'/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/'
#'/FULL_domain/LGS_no_ALM/'
#'/FULL_domain/LGS_ALM_initial/'
#'/FULL_domain/LGS_ALM_FDC_0p10/'
#'/FULL_domain/LGS_ALM_FDC_0p24/'
# '/FULL_domain/LGS_ALM_FDC_0p30/' 


folder_case = folder + '/FULL_domain/LGS_ALM_FDC_0p10/'



SMD_to_read = 'SMD_FW' # 'SMD', 'SMD_FW'

#SPS
format_SPS = '--^k'
label_SPS = r'$\mathrm{UG}100\_\mathrm{DX}10$'
SMD_from_SPS = 80.2

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


# experimental SMD
SMD_expe = 31

# estimate expe errors
#error_SMD  = 0.26
#error_flux = 0.37
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



#%% get data

df_SMD_store_var_old = pd.read_csv(folder_case + '/store_variables/SMD_evolution.csv')
df_SMD_store_var_new = pd.read_csv(folder_case + '/store_variables_new/SMD_evolution.csv')



x_store_old = df_SMD_store_var_old['x'].values
x_store_old = np.insert(x_store_old,0,5)
SMD_store_var_old = df_SMD_store_var_old[SMD_to_read].values
SMD_store_var_old = np.insert(SMD_store_var_old,0,SMD_from_SPS)

x_store_new = df_SMD_store_var_new['x'].values
x_store_new = np.insert(x_store_new,0,5)
SMD_store_var_new = df_SMD_store_var_new[SMD_to_read].values
SMD_store_var_new = np.insert(SMD_store_var_new,0,SMD_from_SPS)





#%% plot


figsize_SMD_evol_along_x = (FFIG*30,FFIG*12)


x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)

plt.figure(figsize=figsize_SMD_evol_along_x)
# Experimentql result
plt.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
plt.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
# LGS results
plt.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)
plt.plot(x_store_old,SMD_store_var_old, 'k', label=r'OLD~folder')
plt.plot(x_store_new,SMD_store_var_new, '--b', label=r'NEW~folder')
plt.xlabel(label_x) 
plt.ylabel(label_SMD) 
#plt.legend(loc='best', ncol=1)
plt.legend(bbox_to_anchor=(1.0,0.75))
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks(x_ticks_SMD_evol)
plt.yticks(y_ticks_SMD_evol)
plt.xlim(4,81)#(plot_bounds[0])
#plt.xlim(13.5,16.5)
plt.ylim(00,82)#(plot_bounds[1])
#plt.savefig(folder_manuscript+'SMD_vs_x_gaseous_BCs_comparison.pdf')
plt.show()
plt.close()      


