
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""





import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import pickle
import matplotlib.pyplot as plt
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
plt.rcParams['legend.fontsize']  = 50*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 20*FFIG #20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*18,FFIG*13)

label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'
label_dx   = r'$\Delta x_\mathrm{atom}~[\mathrm{mm}]$'



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
folder_dx_atom = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST/param_dx_atom'





formats = {'APTE':'-ok', 'TAB':'-^b', 'ETAB':'-*r'}

SMD_from_SPS = 75

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



#%% fixed data

# Expe data
SMD_expe = 31
var_SMD_expe = 0.24 # in %

# Apte simulation data
SMD_dx00    = 20
SMD_FW_dx00 = 20

#%% get SMD evol and data to plot

label_APTE = r'$\mathrm{Goro}$'
label_TAB = r'$\mathrm{TAB}$'
label_ETAB = r'$\mathrm{ETAB}$'

dx_ticks = np.arange(0,21,2)
# 

df_SMD_evol_with_dx_atom = pd.read_csv(folder_dx_atom+'/SMD_evol_with_dx_atom.csv')

dx_values  = df_SMD_evol_with_dx_atom['x'].values
dx_values = np.insert(dx_values,0,0)
SMD_values = df_SMD_evol_with_dx_atom['SMD'].values
SMD_values = np.insert(SMD_values,0,SMD_dx00)
SMD_FW_values = df_SMD_evol_with_dx_atom['SMD_FW'].values
SMD_FW_values = np.insert(SMD_FW_values,0,SMD_FW_dx00)


# expe
expe_SMD_to_plot = [SMD_expe]*len(dx_values)
min_expe_SMD_to_plot = [SMD_expe*(1-var_SMD_expe)]*len(dx_values)
max_expe_SMD_to_plot = [SMD_expe*(1+var_SMD_expe)]*len(dx_values)


#%% plot

plt.figure(figsize=figsize_)
plt.fill_between(dx_values, min_expe_SMD_to_plot, 
                 max_expe_SMD_to_plot, alpha=0.1, facecolor='black')
plt.plot(dx_values,expe_SMD_to_plot,'--k',label=label_expe)
plt.plot(dx_values,SMD_values, '-ok', label=r'$\mathrm{Simulations}$')
#plt.plot(dx_values,SMD_values, formats['TAB'], label=r'SMD-FW')
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel(label_dx) 
plt.ylabel(label_SMD) 
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks(dx_ticks)
plt.xlim(-0.5,20.5)#(plot_bounds[0])
plt.ylim(10,40)#(plot_bounds[1])
plt.savefig(folder_manuscript+'SMD_vs_dx_atom.pdf')
plt.show()
plt.close()   

