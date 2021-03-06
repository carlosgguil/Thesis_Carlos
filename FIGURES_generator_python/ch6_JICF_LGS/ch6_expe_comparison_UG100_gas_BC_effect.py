
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

parent_dir = folder+"LGS_sprays_beginning_CERFACS"
#dirs       = ["xInj10mm/ALM_no_second_no" , "xInj10mm/ALM_no_second_yes", 
#              "xInj10mm/ALM_yes_second_no", "xInj10mm/ALM_yes_second_yes"]

dirs       = ["dx10_x05mm_wRMS_no_ALM","dx10_x05mm_wRMS" , "custom_dx10_x02mm_wRMS"]
sols_dirs_name  = None
filename        = "vol_dist_coarse"  # Only applies if 'loadDropletsDistr = False'
sampling_planes = ['x = 80 mm']


label_1 = r'$\mathrm{No~pert.}$'
label_2 = r'$\mathrm{ALM}$'
label_3 = r'$\mathrm{Custom}$'


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


#%% Plot maps
sprPlot.plot_map(grids_list[0][0], sprays_list[0][0].name, 'Q',
                                 PLOT_GRID = False, PLOT_PIXELS = False)


#%% Plot the PDFs

# Get fit to experimental distribution by Jaegle's
D_values_exp = np.linspace(12.35, 71.63, 100)
D_mean   = 27.31
D_std    = 8.35
D_mean_lg = np.log(D_mean**2/(np.sqrt(D_mean**2 + D_std**2)))
D_std_lg  = np.sqrt(np.log(1 + D_std**2/D_mean**2))
f0_exp = lognormal(D_values_exp, D_mean_lg, D_std_lg)


# Plot PDFs
plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title(f"Size distribution")
plt.plot(D_values_exp, f0_exp, 'r', label='Experimental distribution')
plt.plot(sprays_list[0][0].spaceDiam, sprays_list[0][0].kde.PDF_f0, 
         color='C0', label=label_1)
plt.plot(sprays_list[1][0].spaceDiam, sprays_list[1][0].kde.PDF_f0, 
         'k--', label=label_2)
plt.legend(loc='upper right')
plt.xlabel(r'Diameter [$\mu$m]')
plt.ylabel(r'Probability [$\mu$m$^{-1}$]')
plt.xlim(0,150)
plt.show()
plt.close()



#%% Average along y, to show with respecto to vertical distance z


z_loc_01, vol_flux_along_z_01, SMD_along_z_01 = average_along_y(grids_list[0][0])
z_loc_02, vol_flux_along_z_02, SMD_along_z_02 = average_along_y(grids_list[1][0])
z_loc_03, vol_flux_along_z_03, SMD_along_z_03 = average_along_y(grids_list[2][0])
    
plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title(f"Profile of volume flux along z")
plt.plot(flux_z_exp, z_int_exp, 'ks', label=label_expe)
plt.plot(vol_flux_along_z_01, z_loc_01, 'k', label=label_1)
plt.plot(vol_flux_along_z_02, z_loc_02, 'b', label=label_2)
plt.plot(vol_flux_along_z_03, z_loc_03, 'r', label=label_3)
plt.legend(loc='best')
plt.xlabel(x_label_ql)
plt.ylabel(y_label_z)
plt.show()
plt.close()



plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title(f"Profile of SMD along z")
plt.plot(SMD_z_exp, z_int_exp, 'ks', label=r'Experimental data')
plt.plot(SMD_along_z_01, z_loc_01, 'k', label=label_1)
plt.plot(SMD_along_z_02, z_loc_02, 'b', label=label_2)
plt.plot(SMD_along_z_03, z_loc_03, 'r', label=label_3)
#plt.legend(loc='best')
plt.xlabel(x_label_SMD)
plt.ylabel(y_label_z)
plt.show()
plt.close()
