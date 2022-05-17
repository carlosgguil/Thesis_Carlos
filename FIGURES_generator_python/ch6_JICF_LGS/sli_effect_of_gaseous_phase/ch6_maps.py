
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

figsize_along_z  = (FFIG*10,FFIG*15)

label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
#label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_ql  = r'$q_l$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
#label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'


SAVEFIG = False


SMD_lim_along_z = (10,40)
ql_lim_along_z = (0,2.5)
z_lim = (0,30)
SMD_lim_along_y = (15,35)
ql_lim_along_y = (0,2.5)
y_lim = (-12.5,12.5)

# for maps
flux_levels_ = np.linspace(0,6.0,13)
SMD_levels_ = np.linspace(8,40,15)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/maps/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'


store_variables_folder = 'store_variables' # 'store_variables' 'prev_store_variables'

folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_full_no_ALM  = folder + '/FULL_domain/LGS_no_ALM/'+store_variables_folder+'/'
folder_full_ALM_initial = folder + '/FULL_domain/LGS_ALM_initial/'+store_variables_folder+'/'
folder_full_FDC_0p24 = folder + '/FULL_domain/LGS_ALM_FDC_0p24/'+store_variables_folder+'/'
folder_full_FDC_0p30 = folder + '/FULL_domain/LGS_ALM_FDC_0p30/'+store_variables_folder+'/'


label_expe  = r'$\mathrm{Experiments}$'
label_APTE = r'$\mathrm{Prescribed}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'
label_full_ALM_FDC_0p10 = r'$\mathrm{ALM~tilted}$'
label_full_ALM_FDC_0p24 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p30 = r'$\mathrm{ALM~forced}$'


folders = [folder_APTE,
           folder_full_no_ALM,
           folder_full_ALM_initial,
           folder_full_FDC_0p24,
           folder_full_FDC_0p30]
cases = ['prescribed',
         'no_ALM',
         'ALM_initial',
         'ALM_FDC_0p24',
         'ALM_FDC_0p30']

labels = [label_APTE,
          label_full_no_ALM,
          label_full_ALM_initial,
          label_full_ALM_FDC_0p24,
          label_full_ALM_FDC_0p30]


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








#%% plot expe maps

AR = 1.4
factor = 12

figsize_ = (FFIG*factor,FFIG*factor*AR)
figsize_expe =  (FFIG*factor*1.2,FFIG*factor*AR)


yticks_ = [0,5,10,15,20,25,30]

# expe flux
plt.figure(figsize=figsize_expe)
plt.contourf(expe_yy_values, expe_zz_values, expe_flux_values, levels = flux_levels_, cmap='binary')
#plt.colorbar(format = '$%.1f$', label = label_ql)
contour = plt.contour(expe_yy_values, expe_zz_values, expe_flux_values, 
                      levels = flux_levels_, colors= 'k', linewidths = 2*FFIG, label = label_ql)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xlabel(label_y) 
plt.ylabel(label_z) 
#plt.axis('off')
plt.title(label_expe)
plt.xticks([-10, -5, 0, 5, 10])
#plt.xticks([])
plt.yticks(yticks_)
plt.xlim(y_lim)#(plot_bounds[0])
plt.ylim(z_lim)#(plot_bounds[1])
plt.tight_layout()
if SAVEFIG:
    plt.savefig(folder_manuscript + 'expe_flux.png')
#plt.yticks(y_ticks_)
plt.show()
plt.close()           

# expe SMD
plt.figure(figsize=figsize_expe)
plt.contourf(expe_yy_values, expe_zz_values, expe_SMD_values, levels = SMD_levels_, cmap='binary')
#plt.colorbar(format = '$%.1f$', label = label_SMD)
contour = plt.contour(expe_yy_values, expe_zz_values, expe_SMD_values, 
                      levels = SMD_levels_, colors= 'k', linewidths = 2*FFIG, label = label_ql)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel(label_y) 
plt.ylabel(label_z) 
#plt.axis('off')
#plt.title(label_expe)
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks(yticks_)
plt.xlim(y_lim)#(plot_bounds[0])
plt.ylim(z_lim)#(plot_bounds[1])
plt.tight_layout()
if SAVEFIG:
    plt.savefig(folder_manuscript + 'expe_SMD.png')
#plt.yticks(y_ticks_)
plt.show()
plt.close()           



#%% plot numerical maps


figsize_cbar = (FFIG*factor*1.375,FFIG*factor*AR)

for i in range(len(grids_list)):
    case = cases[i]
    grid = grids_list[i][0]
    
    if case == 'ALM_initial' or case == 'ALM_FDC_0p30':
        figsize_numerical = figsize_cbar
    else:
        figsize_numerical = figsize_


    # flux
    plt.figure(figsize=figsize_numerical)
    plt.contourf(grid.yy_center, grid.zz_center, grid.map_vol_flux*1e2, levels = flux_levels_, cmap='binary')

    if case == 'ALM_initial' or case == 'ALM_FDC_0p30':
        #plt.colorbar(format = '%.1f',ticks=levels_)
        plt.colorbar(format = '$%.1f$', label = label_ql)
        
    contour = plt.contour(grid.yy_center, grid.zz_center, grid.map_vol_flux*1e2, 
                          levels = flux_levels_[1:], colors= 'k', linewidths = 2*FFIG)
    #plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
    #plt.xlabel(label_y) 
    #plt.ylabel(label_z) 
    plt.yticks([])
    plt.title(labels[i])
    plt.tight_layout()
    #plt.axis('off')
    plt.xticks([-10, -5, 0, 5, 10])
    #plt.xticks([])
    plt.xlim(y_lim)#(plot_bounds[0])
    plt.ylim(z_lim)#(plot_bounds[1])
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(folder_manuscript+cases[i]+'_flux.png')
    plt.show()
    plt.close()      
    
    
    
    # SMD
    plt.figure(figsize=figsize_numerical)
    plt.contourf(grid.yy_center, grid.zz_center, grid.map_SMD, levels = SMD_levels_, cmap='binary')
    
    if case == 'ALM_initial' or case == 'ALM_FDC_0p30':
        #plt.colorbar(format = '%.1f',ticks=levels_)
        plt.colorbar(format = '$%.1f$', label = label_SMD)
    contour = plt.contour(grid.yy_center, grid.zz_center, grid.map_SMD, 
                          levels = SMD_levels_[1:], colors= 'k', linewidths = 2*FFIG)
    #plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
    plt.xlabel(label_y) 
    #plt.ylabel(label_z) 
    plt.yticks([])
    #plt.title(labels[i])
    #plt.axis('off')
    #plt.title('$\mathrm{Jaegle~(2008)}$')
    plt.xticks([-10, -5, 0, 5, 10])
    plt.xlim(y_lim)#(plot_bounds[0])
    plt.ylim(z_lim)#(plot_bounds[1])
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(folder_manuscript+cases[i]+'_SMD.png')
    plt.show()
    plt.close()      
