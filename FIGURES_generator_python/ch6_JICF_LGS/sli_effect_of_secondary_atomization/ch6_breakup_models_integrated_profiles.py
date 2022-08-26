
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
from sprPost_functions import get_grids_common_boundaries, pickle_load
import sprPost_plot as sprPlot


# Change size of figures 
FFIG = 0.5
#mpl.rcParams['font.size'] = 40*fPic
plt.rcParams['xtick.labelsize']  = 70*FFIG
plt.rcParams['ytick.labelsize']  = 70*FFIG
plt.rcParams['axes.labelsize']   = 70*FFIG
plt.rcParams['axes.labelpad']    = 30*FFIG
plt.rcParams['axes.titlesize']   = 80*FFIG
plt.rcParams['legend.fontsize']  = 40*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 20*FFIG #20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True

figsize_along_y = (FFIG*18,FFIG*13)
figsize_along_z  = (FFIG*10,FFIG*15)

label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
#label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
#label_ql  = r'$q_l$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_ql  = r'$\left\langle q_l \right\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
#label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD = r'$\left\langle SMD \right\rangle$ [$\mu\mathrm{m}$]'

SAVEFIG = False




# for maps
flux_levels_ = np.linspace(0,6.0,13)
SMD_levels_ = np.linspace(8,40,15)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_breakup_model/profiles/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'


store_variables_folder = 'store_variables' # 'store_variables' 'prev_store_variables'


folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_TAB  = folder + '/param_breakup_model/TAB_u_vw_LN/store_variables/'
folder_ETAB = folder + '/param_breakup_model/ETAB_u_vw_LN/store_variables/'


label_expe  = r'$\mathrm{Experiments}$'
label_APTE = r'$\mathrm{Gorokhovski}$'
label_TAB = r'$\mathrm{TAB}$'
label_ETAB = r'$\mathrm{ETAB}$'


folders = [folder_APTE,
           folder_TAB,
           folder_ETAB]

cases = ['goro',
         'TAB',
         'ETAB']

labels_ = [label_APTE,
          label_TAB,
          label_ETAB]


formats_ = {'goro':'-ok', 'TAB':'-^b', 'ETAB':'-*r'}
formats_ = {'goro':'-k', 'TAB':'-b', 'ETAB':'-r'}

color_expe = 'black'
linewidth_expe = 10*FFIG
format_expe = '-s'

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


#%% read numerical results






sprays_list = [[] for i in range(len(folders))]
grids_list = [[] for i in range(len(folders))]

for i in range(len(folders)):
    dir_i   = folders[i]
    spray = pickle_load(dir_i + 'sprays_list_x=80mm')
    grid  = pickle_load(dir_i + 'grids_list_x=80mm')
    
    sprays_list[i].append(spray)
    grids_list[i].append(grid)




#%% Average along y and z


z_loc = []; vol_flux_along_z = []; SMD_along_z = []
y_loc = []; vol_flux_along_y = []; SMD_along_y = []
for i in range(len(grids_list)):
    # averaged along y
    z_loc_i, vol_flux_i, SMD_i = average_along_y(grids_list[i][0])
    z_loc.append(z_loc_i)
    vol_flux_along_z.append(vol_flux_i)
    SMD_along_z.append(SMD_i)
    
    # averaged along z
    y_loc_i, vol_flux_i, SMD_i = average_along_z(grids_list[i][0])
    y_loc.append(y_loc_i)
    vol_flux_along_y.append(vol_flux_i)
    SMD_along_y.append(SMD_i)





#%% Plot profiles along z
 
SMD_lim_along_z = (7,43)
SMD_ticks_along_z = [10,20,30,40]
z_ticks = [0,5,10,15,20,25,30]
ql_lim_along_z = (0,2.7)
ql_ticks_along_z = [0,1,2]
z_lim = (0,30)

factor = 1.1
figsize_flux_along_z  = (FFIG*10*factor,FFIG*17.3*factor)
figsize_SMD_along_z  = (FFIG*8*factor,FFIG*17.0*factor)


   
plt.figure(figsize=figsize_flux_along_z)
plt.plot(flux_z_exp, z_int_exp, format_expe, color=color_expe,label=label_expe,
         linewidth = linewidth_expe)
plt.errorbar(flux_z_exp, z_int_exp, xerr=error_q_z_expe, color=color_expe, fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(grids_list)):
    case_i = cases[i]
    plt.plot(vol_flux_along_z[i], z_loc[i], formats_[case_i], label=labels_[i])
plt.legend(loc='best')
plt.xlabel(label_ql)
plt.xlim(ql_lim_along_z)
plt.xticks(ql_ticks_along_z)
plt.ylabel(label_z)
plt.ylim(z_lim)
plt.yticks(z_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript + 'flux_along_z.pdf')
plt.show()
plt.close()



plt.figure(figsize=figsize_SMD_along_z)
plt.plot(SMD_z_exp, z_int_exp, format_expe, color=color_expe, label=label_expe,
         linewidth = linewidth_expe)
plt.errorbar(SMD_z_exp, z_int_exp, xerr=error_SMD_z_expe, color=color_expe, fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(grids_list)):
    case_i = cases[i]
    plt.plot(SMD_along_z[i], z_loc[i], formats_[case_i], label=labels_[i])
#plt.legend(loc='best')
plt.xlabel(label_SMD)
plt.xlim(SMD_lim_along_z)
plt.xticks(SMD_ticks_along_z)

#plt.ylabel(label_z)
plt.ylabel('')
plt.ylim(z_lim)
plt.yticks(z_ticks)
ax = plt.gca()
ax.yaxis.set_ticklabels([])
#plt.yticks(z_ticks)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript + 'SMD_along_z.pdf')
plt.show()
plt.close()



#%% Plot profiles along y

(FFIG*18,FFIG*13)
factor = 1.1
figsize_flux_along_y  = (FFIG*17*factor,FFIG*8*factor)#(FFIG*10,FFIG*15)
figsize_SMD_along_y  = (FFIG*17*factor,FFIG*10*factor) #(FFIG*8,FFIG*14.7)


SMD_lim_along_y = (0,38)
SMD_ticks_along_y = [0,10,20,30] #[10,15,20,25,30,35]
ql_lim_along_y = (0,3.2)
ql_ticks_along_y = [0, 1, 2, 3]#[0,0.5,1,1.5,2,2.5,3]
y_lim = (-12.5,12.5)
y_ticks = [-10,-5,0,5,10]

plt.figure(figsize=figsize_flux_along_y)
plt.plot(y_int_exp, flux_y_exp, format_expe, color=color_expe, label=label_expe,
         linewidth = linewidth_expe)
plt.errorbar(y_int_exp, flux_y_exp, yerr=error_q_y_expe, color=color_expe, fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(grids_list)):
    case_i = cases[i]
    plt.plot(y_loc[i], vol_flux_along_y[i], formats_[case_i], label=labels_[i])
#plt.legend(loc='best')
#plt.xlabel(label_y)
plt.xlabel('')
plt.xlim(y_lim)
plt.xticks(y_ticks)
ax = plt.gca()
ax.xaxis.set_ticklabels([])
plt.ylabel(label_ql, labelpad=57*FFIG)
plt.ylim(ql_lim_along_y)
plt.yticks(ql_ticks_along_y)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript + 'flux_along_y.pdf')
plt.show()
plt.close()


plt.figure(figsize=figsize_SMD_along_y)
plt.plot(y_int_exp, SMD_y_exp, format_expe, color=color_expe, label=label_expe,
         linewidth = linewidth_expe)
plt.errorbar(y_int_exp, SMD_y_exp, yerr=error_SMD_y_expe, color=color_expe, fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(grids_list)):
    case_i = cases[i]
    plt.plot(y_loc[i], SMD_along_y[i], formats_[case_i], label=labels_[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.xlim(y_lim)
plt.xticks(y_ticks)
plt.ylabel(label_SMD)
plt.ylim(SMD_lim_along_y)
plt.yticks(SMD_ticks_along_y)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript + 'SMD_along_y.pdf')
plt.show()
plt.close()

#%% Get SMDs and errors

print('----- SMDs ------')
for i in range(len(sprays_list)):
    sp = sprays_list[i][0]
    grid = grids_list[i][0]
    SMD = sp.SMD
    err_SMD = (SMD - SMD_expe)/SMD_expe*100
    SMD_FW = get_SMD_flux_weighted(grid)
    err_SMD_FW = (SMD_FW - SMD_expe)/SMD_expe*100
    print('Case '+labels_[i]+':')
    print(f' Arithmetic: {SMD:.2f} ({err_SMD:.2f} %), flux_weighted = {SMD_FW:.2f} ({err_SMD_FW:.2f} %)')
    print(f'       Q: {sp.Q*1e9}')
    