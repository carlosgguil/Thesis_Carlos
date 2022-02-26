
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

    


import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions_expe_comparison import average_along_y, average_along_z
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
plt.rcParams['lines.markersize'] = 20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*18,FFIG*13)

folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/'
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/expe_validation_set_4/'


label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'

labels_ = [r'$\mathrm{Simulation~low~We}$']


formats_ = ['b']
SMD_lim_along_z = (20,45)
ql_lim_along_z = (0,2)
z_lim = (0,30)
SMD_lim_along_y = (25,40)
ql_lim_along_y = (0,2.5)
y_lim = (-12.5,12.5)



#%% Experimental data and simulation parameters (do not touch)
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'
data_int_y_exp = pd.read_csv(folder_expe + '2310_01_u75_data_integrated_y_exp.csv')
data_int_z_exp = pd.read_csv(folder_expe + '2310_01_u75_data_integrated_z_exp.csv')

z_int_exp  = data_int_y_exp['z_values']
flux_z_exp = data_int_y_exp['flux_z_exp']
SMD_z_exp  = data_int_y_exp['SMD_z_exp']

y_int_exp  = data_int_z_exp['y_values']
flux_y_exp = data_int_z_exp['flux_y_exp']
SMD_y_exp  = data_int_z_exp['SMD_y_exp']


params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']











#%% CASE TO CHECK                       

parent_dir = folder+"LGS_sprays_2nd_op"
#dirs       = ["xInj10mm/ALM_no_second_no" , "xInj10mm/ALM_no_second_yes", 
#              "xInj10mm/ALM_yes_second_no", "xInj10mm/ALM_yes_second_yes"]

dirs       = ["dx10_x05mm_wRMS"]
sols_dirs_name  = None
filename        = "vol_dist_coarse"  # Only applies if 'loadDropletsDistr = False'
sampling_planes = ['x = 80 mm']





#%% LOAD THE SPRAYS AND GRIDS

dirs = [parent_dir+'/'+d for d in dirs]
sprays_list = get_sprays_list(True, sampling_planes, dirs, filename,
                              params_simulation,
                              sols_dirs_name = '.',
                              D_outlier = 300000)

dirs_grid = [ [d] for d in dirs]
grids_list = get_discrete_spray(True, sprays_list, [8]*2, 
                                None, params_simulation ,
                                DIR = dirs_grid)

common_bounds = get_grids_common_boundaries(grids_list)





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
    
#%% Plot along z
    
plt.figure(figsize=figsize_)
plt.plot(flux_z_exp, z_int_exp, '-ks', label=label_expe)
for i in range(len(grids_list)):
    plt.plot(vol_flux_along_z[i], z_loc[i], formats_[i], label=labels_[i])
plt.legend(loc='best')
plt.xlabel(label_ql)
plt.ylabel(label_z)
plt.xlim(ql_lim_along_z)
plt.ylim(z_lim)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_fluxes_along_z.pdf')
plt.show()
plt.close()



plt.figure(figsize=figsize_)
plt.plot(SMD_z_exp, z_int_exp, '-ks', label=label_expe)
for i in range(len(grids_list)):
    plt.plot(SMD_along_z[i], z_loc[i], formats_[i], label=labels_[i])
#plt.legend(loc='best')
plt.xlabel(label_SMD)
plt.ylabel(label_z)
plt.xlim(SMD_lim_along_z)
plt.ylim(z_lim)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_SMD_along_z.pdf')
plt.show()
plt.close()

#%% Plot along y

plt.figure(figsize=figsize_)
plt.plot(y_int_exp, flux_y_exp, '-ks', label=label_expe)
for i in range(len(grids_list)):
    plt.plot(y_loc[i], vol_flux_along_y[i], formats_[i], label=labels_[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_SMD)
plt.ylim(y_lim)
plt.ylim(ql_lim_along_y)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_fluxes_along_y.pdf')
plt.show()
plt.close()


plt.figure(figsize=figsize_)
plt.plot(y_int_exp, SMD_y_exp, '-ks', label=label_expe)
for i in range(len(grids_list)):
    plt.plot(y_loc[i], SMD_along_y[i], formats_[i], label=labels_[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_SMD)
plt.xlim(y_lim)
plt.ylim(SMD_lim_along_y)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_SMD_along_y.pdf')
plt.show()
plt.close()
