# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:31:48 2020

@author: d601630
"""


import sys
sys.path.append('C:/Users/d601630/Documents/GitHub/sprays_tools/jicf_penetration')
sys.path.append('C:/Users/d601630/Desktop/Ongoing/JICF_trajectories/trajectories_SPS')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trajectory_calculation_functions as trj
from functions_methods import get_mean_trajectory_sweep



# Change size of figures if wished
FFIG = 1.0


folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_trajectories/'
folder = 'C:/Users/d601630/Desktop/Ongoing/JICF_trajectories/trajectories_SPS/data_last_trajectories_U_inlets'
# Physical parameters (lists, one value per figure)
q             = 6 #Becker 6, Ragucci 14.2  # Kinetic energy ratio [-]
d_inj         = 0.45  # Becker 0.45, Ragucci 0.5   # Injector diameter [mm]



# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

##########################################################

x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')


op1 = 'q6uG100'

z_max = 15.0


#%% Load and process trajectories 

# OP1, dx = 20 µm
directory_op1_dx20     = folder+'/'+op1+'/dx20_LS_REINIT_FALSE_U_user_defined'
data_op1_dx20_method_a = pd.read_csv(directory_op1_dx20+'/method_a_data_trajectory.csv')
L2_op1_dx20_method_a   = pd.read_csv(directory_op1_dx20+'/method_a_data_L2.csv')
data_op1_dx20_method_b = pd.read_csv(directory_op1_dx20+'/method_b_data_trajectory.csv')
L2_op1_dx20_method_b   = pd.read_csv(directory_op1_dx20+'/method_b_data_L2.csv')
data_op1_dx20_method_c = pd.read_csv(directory_op1_dx20+'/method_c_data_trajectory.csv')
L2_op1_dx20_method_c   = pd.read_csv(directory_op1_dx20+'/method_c_data_L2.csv')
data_op1_dx20_method_d = pd.read_csv(directory_op1_dx20+'/method_d_data_trajectory.csv')
L2_op1_dx20_method_d   = pd.read_csv(directory_op1_dx20+'/method_d_data_L2.csv')

#%% Get errors with axial location
becker_corr_method_a = trj.trajectory_vertical(data_op1_dx20_method_a['xD'].values*d_inj, d_inj)
becker_corr_method_a.get_trajectory(d_inj, q, correlation='becker')
error_a = becker_corr_method_a.zD_mean - data_op1_dx20_method_a['zD'].values


becker_corr_method_b = trj.trajectory_vertical(data_op1_dx20_method_b['xD'].values*d_inj, d_inj)
becker_corr_method_b.get_trajectory(d_inj, q, correlation='becker')
error_b = becker_corr_method_b.zD_mean - data_op1_dx20_method_b['zD'].values


becker_corr_method_c = trj.trajectory_vertical(data_op1_dx20_method_c['xD'].values*d_inj, d_inj)
becker_corr_method_c.get_trajectory(d_inj, q, correlation='becker')
error_c = becker_corr_method_c.zD_mean - data_op1_dx20_method_c['zD'].values


becker_corr_method_d = trj.trajectory_vertical(data_op1_dx20_method_d['xD'].values*d_inj, d_inj)
becker_corr_method_d.get_trajectory(d_inj, q, correlation='becker')
error_d = becker_corr_method_d.zD_mean - data_op1_dx20_method_d['zD'].values

#%% OP1, dx = 20 µm all methods together
title = r'OP1'

# Trajectory
plt.figure(figsize=(FFIG*22,FFIG*15))
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op1_dx20_method_a['xD'], data_op1_dx20_method_a['zD'], 'r', label = r'INST\_NM',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_b['xD'], data_op1_dx20_method_b['zD'], 'b', label = r'INST\_M',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_c['xD'], data_op1_dx20_method_c['zD'], 'g', label = r'MEAN\_GRAD',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_d['xD'], data_op1_dx20_method_d['zD'], 'y', label = r'MEAN\_CONT',linewidth=4*FFIG)
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
plt.ylim(becker_corr.plotD_limits[1][0], z_max)#becker_corr.plotD_limits[1][1])
plt.grid()
plt.legend(loc='lower right', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_trajectories_q6uG100.pdf')
plt.savefig(folder_manuscript+'methods_comparison_trajectories_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# L2 error
plt.figure(figsize=(FFIG*22,FFIG*15))
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.plot(L2_op1_dx20_method_a['t'], L2_op1_dx20_method_a['L2'], 'r', label = r'INST\_NN',linewidth=4*FFIG)
plt.plot(L2_op1_dx20_method_b['t'], L2_op1_dx20_method_b['L2'], 'b', label = r'INST\_M',linewidth=4*FFIG)
plt.plot(L2_op1_dx20_method_c['t'], L2_op1_dx20_method_c['L2'], 'g', label = r'MEAN\_GRAD',linewidth=4*FFIG)
plt.plot(L2_op1_dx20_method_d['t'], L2_op1_dx20_method_d['L2'], 'y', label = r'MEAN\_CONT',linewidth=4*FFIG)
plt.xlabel(r'$\mathrm{Iteration}$')
plt.ylabel(r'$L_2$')
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_L2_evolution_q6uG100.pdf')
plt.savefig(folder_manuscript+'methods_comparison_L2_evolution_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# Trajectory difference
plt.figure(figsize=(FFIG*22,FFIG*15))
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op1_dx20_method_a['xD'], error_a, 'r', label = r'INST\_NM',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_b['xD'], error_b, 'b', label = r'INST\_M',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_c['xD'], error_c, 'g', label = r'MEAN\_GRAD',linewidth=4*FFIG)
plt.plot(data_op1_dx20_method_d['xD'], error_d, 'y', label = r'MEAN\_CONT',linewidth=4*FFIG)
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\mathrm{Error}$')
plt.xlim(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
plt.ylim(-0.5, 4)#becker_corr.plotD_limits[1][1])
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()




