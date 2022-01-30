
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""
def average_along_y(grid):
    z_locations   = np.zeros(grid.grid_size[1])
    vol_flux_along_z = np.zeros(grid.grid_size[1])
    SMD_along_z = np.zeros(grid.grid_size[1])
    for m in range(grid.grid_size[1]):
        
        ly = 0
        vol_flux_x_dy = 0
        SMD_x_dy = 0
        for n in range(grid.grid_size[0]):
            vol_flux_current = grid.map_vol_flux.data[m][n]
            SMD_current = grid.map_SMD.data[m][n]
            if np.isnan(vol_flux_current) or np.isnan(SMD_current):
                continue
            vol_flux_x_dy += vol_flux_current*grid.dy
            SMD_x_dy += SMD_current*vol_flux_current*grid.dy
            ly += grid.dy
          
        z_locations[m] = grid.zz_center[m][0]
        vol_flux_along_z[m] = vol_flux_x_dy/ly
        SMD_along_z[m] = SMD_x_dy/ly/vol_flux_along_z[m]
    
    #vol_flux_along_z = normalize(z_locations, vol_flux_along_z)
    
    return z_locations, vol_flux_along_z*100, SMD_along_z



def lognormal(x, mean, std):

    # If the following error appears, it is due to the design space of the optimization algorithm
    
    f = np.zeros(len(x))
    for i in range(len(x)):
        #f[i] = 1/(x[i]*np.log(sigmag)*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(x[i]/mean)/np.log(sigmag))**2)
        f[i] = 1/(x[i]*std*np.sqrt(2*np.pi))*np.exp(-0.5*(np.log(x[i]) - mean)**2/std**2)
    return f



def normalize(x, signal):
    
    A = 0
    for i in range(len(signal)-1):
        area_i = (signal[i] + signal[i+1])*(x[i+1]-x[i])/2
        A += area_i
    
    signal = signal/A
    
    return signal
    


import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/'
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_trajectories/'



y_label_z   = r'$z~[\mathrm{mm}]$'
x_label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
x_label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'




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


params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']











#%% CASE TO CHECK                       

parent_dir = folder+"LGS_sprays_custom_inlet_v02"
#dirs       = ["xInj10mm/ALM_no_second_no" , "xInj10mm/ALM_no_second_yes", 
#              "xInj10mm/ALM_yes_second_no", "xInj10mm/ALM_yes_second_yes"]

dirs       = ["dx10_x02mm_wRMS" , 
              "dx10_x02mm_wRMS_u_rel_perc_200_TRUE",
              "dx10_x02mm_wRMS_u_rel_perc_100_TRUE",
              "dx10_x02mm_wRMS_u_rel_perc_075_TRUE",
              "dx10_x02mm_wRMS_u_rel_perc_050_TRUE",
              "dx10_x02mm_wRMS_u_rel_perc_025_TRUE"]


sols_dirs_name  = None
filename        = "vol_dist_coarse"  # Only applies if 'loadDropletsDistr = False'
sampling_planes = ['x = 80 mm']


labels_ = [r'$\alpha = \infty$',  
           r'$\alpha = 300~\%$',
           r'$\alpha = 200~\%$',
           r'$\alpha = 100~\%$',
           r'$\alpha = 75~\%$',
           r'$\alpha = 50~\%$',
           r'$\alpha = 25~\%$']

format_ = ['--k', 'k', 'b', 'r', 'g', 'y']

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




#%% Average along y, to show with respecto to vertical distance z

z_loc = []; vol_flux_along_z = []; SMD_along_z = []
for i in range(len(grids_list)):    
    z_loc_i, vol_flux_along_z_i, SMD_along_z_i = average_along_y(grids_list[i][0])
    z_loc.append(z_loc_i)
    vol_flux_along_z.append(vol_flux_along_z_i)
    SMD_along_z.append(SMD_along_z_i)

    
plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title(f"Profile of volume flux along z")
plt.plot(flux_z_exp, z_int_exp, 'ks', label=label_expe)
for i in range(len(grids_list)):    
    plt.plot(vol_flux_along_z[i], z_loc[i], format_[i], label=labels_[i])
plt.legend(loc='best')
plt.grid()
plt.xlabel(x_label_ql)
plt.ylabel(y_label_z)
plt.show()
plt.close()



plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title(f"Profile of SMD along z")
plt.plot(SMD_z_exp, z_int_exp, 'ks', label=r'Experimental data')
for i in range(len(grids_list)):    
    plt.plot(SMD_along_z[i], z_loc[i], format_[i], label=labels_[i])
#plt.legend(loc='best')
plt.grid()
plt.xlabel(x_label_SMD)
plt.ylabel(y_label_z)
plt.show()
plt.close()
