
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

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'



folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_full_no_ALM  = folder + '/FULL_domain/LGS_no_ALM/store_variables/'
folder_full_ALM_initial = folder + '/FULL_domain/LGS_ALM_initial/store_variables/'
folder_full_FDC_0p24 = folder + '/FULL_domain/LGS_ALM_FDC_0p24/store_variables/'
folder_full_FDC_0p30 = folder + '/FULL_domain/LGS_ALM_FDC_0p30/store_variables/'


label_APTE = r'$\mathrm{Prescribed}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'
label_full_ALM_FDC_0p24 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p30 = r'$\mathrm{ALM~modified}$'

formats = {'APTE':'-ok', 
           'full_no_ALM':'-^b', 
           'full_ALM_initial':'-*r',
           'full_ALM_FDC_0p24':'-*y',
           'full_ALM_FDC_0p30':'-*g',}
formats = {'APTE':'-k', 
           'full_no_ALM':'-b', 
           'full_ALM_initial':'-r',
           'full_ALM_FDC_0p24':'-y',
           'full_ALM_FDC_0p30':'-g',}

SMD_to_read = 'SMD' # 'SMD', 'SMD_FW'

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




#%% get data


df_SMD_evol_APTE = pd.read_csv(folder_APTE+'SMD_evolution.csv')
df_SMD_evol_full_no_ALM  = pd.read_csv(folder_full_no_ALM+'SMD_evolution.csv')
df_SMD_evol_full_ALM_initial = pd.read_csv(folder_full_ALM_initial+'SMD_evolution.csv')
df_SMD_evol_full_ALM_FDC_0p24 = pd.read_csv(folder_full_FDC_0p24+'SMD_evolution.csv')
df_SMD_evol_full_ALM_FDC_0p30 = pd.read_csv(folder_full_FDC_0p30+'SMD_evolution.csv')

x_APTE = df_SMD_evol_APTE['x'].values
x_APTE = np.insert(x_APTE,0,5)
SMD_APTE = df_SMD_evol_APTE[SMD_to_read].values
SMD_APTE = np.insert(SMD_APTE,0,SMD_from_SPS)

x_full_no_ALM = df_SMD_evol_full_no_ALM['x'].values
x_full_no_ALM = np.insert(x_full_no_ALM,0,5)
SMD_full_no_ALM = df_SMD_evol_full_no_ALM[SMD_to_read].values
SMD_full_no_ALM = np.insert(SMD_full_no_ALM,0,SMD_from_SPS)

x_full_ALM_initial = df_SMD_evol_full_ALM_initial['x'].values
x_full_ALM_initial = np.insert(x_full_ALM_initial,0,5)
SMD_full_ALM_initial = df_SMD_evol_full_ALM_initial[SMD_to_read].values
SMD_full_ALM_initial = np.insert(SMD_full_ALM_initial,0,SMD_from_SPS)

x_full_ALM_FDC_0p24 = df_SMD_evol_full_ALM_FDC_0p24['x'].values
x_full_ALM_FDC_0p24= np.insert(x_full_ALM_FDC_0p24,0,5)
SMD_full_ALM_FDC_0p24 = df_SMD_evol_full_ALM_FDC_0p24[SMD_to_read].values
SMD_full_ALM_FDC_0p24 = np.insert(SMD_full_ALM_FDC_0p24,0,SMD_from_SPS)

x_full_ALM_FDC_0p30 = df_SMD_evol_full_ALM_FDC_0p30['x'].values
x_full_ALM_FDC_0p30= np.insert(x_full_ALM_FDC_0p30,0,5)
SMD_full_ALM_FDC_0p30 = df_SMD_evol_full_ALM_FDC_0p30[SMD_to_read].values
SMD_full_ALM_FDC_0p30 = np.insert(SMD_full_ALM_FDC_0p30,0,SMD_from_SPS)


#%% plot


label_APTE = r'$\mathrm{Custom}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'





x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)

plt.figure(figsize=figsize_)
plt.plot(x_APTE,SMD_APTE, formats['APTE'], label=label_APTE)
plt.plot(x_full_no_ALM,SMD_full_no_ALM, formats['full_no_ALM'], label=label_full_no_ALM)
plt.plot(x_full_ALM_initial,SMD_full_ALM_initial, formats['full_ALM_initial'], label=label_full_ALM_initial)
plt.plot(x_full_ALM_FDC_0p24,SMD_full_ALM_FDC_0p24, formats['full_ALM_FDC_0p24'], label=label_full_ALM_FDC_0p24)
plt.plot(x_full_ALM_FDC_0p30,SMD_full_ALM_FDC_0p30, formats['full_ALM_FDC_0p30'], label=label_full_ALM_FDC_0p30)
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
plt.savefig(folder_manuscript+'SMD_vs_x_gaseous_BCs_comparison.pdf')
plt.show()
plt.close()      


