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
FFIG = 0.5


folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_trajectories/'
folder = 'C:/Users/d601630/Desktop/Ongoing/JICF_trajectories/trajectories_SPS/data_trajectories'
# Physical parameters (lists, one value per figure)
q             = 6 #Becker 6, Ragucci 14.2  # Kinetic energy ratio [-]
d_inj         = 0.45  # Becker 0.45, Ragucci 0.5   # Injector diameter [mm]



# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
#plt.rcParams['legend.loc']      = 'upper left'
plt.rcParams['text.usetex'] = True

##########################################################

x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')


op1 = 'q6uG100'
op2 = 'q6uG75'

z_max = 15.0


tau_ph_UG100 = 0.026
tau_ph_UG75 = 0.019

dt = 1.5e-3 #ms

#%% Load and process trajectories  (method C)
method = 'method_c'

# OP1, dx = 20 µm
directory_op1_dx20 = folder+'/'+op1+'/dx20'
data_op1_dx20 = pd.read_csv(directory_op1_dx20+'/'+method+'_data_trajectory.csv')
L2_op1_dx20   = pd.read_csv(directory_op1_dx20+'/'+method+'_data_L2.csv')


# OP1, dx = 20 µm no turb.
directory_op1_dx20_no_turb = folder+'/'+op1+'/dx20_no_turbulence'
data_op1_dx20_no_turb = pd.read_csv(directory_op1_dx20_no_turb+'/'+method+'_data_trajectory.csv')
L2_op1_dx20_no_turb   = pd.read_csv(directory_op1_dx20_no_turb+'/'+method+'_data_L2.csv')


# OP1, dx = 10 µm
directory_op1_dx10 = folder+'/'+op1+'/dx10'
data_op1_dx10 = pd.read_csv(directory_op1_dx10+'/'+method+'_data_trajectory.csv')
L2_op1_dx10   = pd.read_csv(directory_op1_dx10+'/'+method+'_data_L2.csv')

# OP2, dx = 20 µm
directory_op2_dx20 = folder+'/'+op2+'/dx20'
data_op2_dx20 = pd.read_csv(directory_op2_dx20+'/'+method+'_data_trajectory.csv')
L2_op2_dx20   = pd.read_csv(directory_op2_dx20+'/'+method+'_data_L2.csv')

# OP2, dx = 10 µm
directory_op2_dx10 = folder+'/'+op2+'/dx10'
data_op2_dx10 = pd.read_csv(directory_op2_dx10+'/'+method+'_data_trajectory.csv')
L2_op2_dx10   = pd.read_csv(directory_op2_dx10+'/'+method+'_data_L2.csv')

#%% Transform  L2 iterations to time
t_L2_op1_dx20 = (L2_op1_dx20['t'].values - 1)*dt/tau_ph_UG100
t_L2_op1_dx20_no_turb = (L2_op1_dx20_no_turb['t'].values - 1)*dt/tau_ph_UG100
t_L2_op1_dx10 = (L2_op1_dx10['t'].values - 1)*dt/tau_ph_UG100

t_L2_op2_dx20 = (L2_op2_dx20['t'].values - 1)*dt/tau_ph_UG75
t_L2_op2_dx10 = (L2_op2_dx10['t'].values - 1)*dt/tau_ph_UG75


#%% Get errors with axial location
becker_corr_op1_dx20 = trj.trajectory_vertical(data_op1_dx20['xD'].values*d_inj, d_inj)
becker_corr_op1_dx20.get_trajectory(d_inj, q, correlation='becker')
error_op1_dx20 = data_op1_dx20['zD'].values - becker_corr_op1_dx20.zD_mean
error_op1_dx20 = error_op1_dx20[1:]
error_op1_dx20 = error_op1_dx20/becker_corr_op1_dx20.zD_mean[1:]*100


becker_corr_op1_dx20_no_turb = trj.trajectory_vertical(data_op1_dx20_no_turb['xD'].values*d_inj, d_inj)
becker_corr_op1_dx20_no_turb.get_trajectory(d_inj, q, correlation='becker')
error_op1_dx20_no_turb = data_op1_dx20_no_turb['zD'].values - becker_corr_op1_dx20_no_turb.zD_mean
error_op1_dx20_no_turb = error_op1_dx20_no_turb[1:]
error_op1_dx20_no_turb = error_op1_dx20_no_turb/becker_corr_op1_dx20_no_turb.zD_mean[1:]*100


becker_corr_op1_dx10 = trj.trajectory_vertical(data_op1_dx10['xD'].values*d_inj, d_inj)
becker_corr_op1_dx10.get_trajectory(d_inj, q, correlation='becker')
error_op1_dx10 = data_op1_dx10['zD'].values -  becker_corr_op1_dx10.zD_mean
error_op1_dx10 = error_op1_dx10[1:]
error_op1_dx10 = error_op1_dx10/becker_corr_op1_dx10.zD_mean[1:]*100


becker_corr_op2_dx20 = trj.trajectory_vertical(data_op2_dx20['xD'].values*d_inj, d_inj)
becker_corr_op2_dx20.get_trajectory(d_inj, q, correlation='becker')
error_op2_dx20 = data_op2_dx20['zD'].values - becker_corr_op2_dx20.zD_mean
error_op2_dx20 = error_op2_dx20[1:]
error_op2_dx20 = error_op2_dx20/becker_corr_op2_dx20.zD_mean[1:]*100


becker_corr_op2_dx10 = trj.trajectory_vertical(data_op2_dx10['xD'].values*d_inj, d_inj)
becker_corr_op2_dx10.get_trajectory(d_inj, q, correlation='becker')
error_op2_dx10 = data_op2_dx10['zD'].values - becker_corr_op2_dx10.zD_mean
error_op2_dx10 = error_op2_dx10[1:]
error_op2_dx10 = error_op2_dx10/becker_corr_op2_dx10.zD_mean[1:]*100



#%% Trajectories 
title = r'OP1'

# OP1
plt.figure(figsize=(FFIG*22,FFIG*15))
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op1_dx10['xD'], data_op1_dx10['zD'], 'b', label = r'UG100\_DX10',linewidth=4*FFIG)
plt.plot(data_op1_dx20['xD'], data_op1_dx20['zD'], 'r', label = r'UG100\_DX20',linewidth=4*FFIG)
#plt.plot(data_op1_dx20_no_turb['xD'], data_op1_dx20_no_turb['zD'], '--r', label = r'UG100\_DX20\_NO\_TURB',linewidth=4*FFIG)
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
plt.ylim(becker_corr.plotD_limits[1][0], 20.0)#becker_corr.plotD_limits[1][1])
plt.xlim(0,10)
plt.ylim(0,12)
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG100.pdf')
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# OP2
plt.figure(figsize=(FFIG*22,FFIG*15))
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op2_dx10['xD'], data_op2_dx10['zD'], 'y', label = r'UG75\_DX10',linewidth=4*FFIG)
plt.plot(data_op2_dx20['xD'], data_op2_dx20['zD'], 'g', label = r'UG75\_DX20',linewidth=4*FFIG)
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
plt.ylim(becker_corr.plotD_limits[1][0], 20.0)#becker_corr.plotD_limits[1][1])
plt.xlim(0,10)
plt.ylim(0,12)
plt.grid()
plt.legend(loc='upper left', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG75.pdf')
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG75.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% L2 error and error

# L2 error
plt.figure(figsize=(FFIG*22,FFIG*15))
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.plot(t_L2_op1_dx10, L2_op1_dx10['L2'], 'b', label = r'UG100\_DX10',linewidth=4*FFIG)
plt.plot(t_L2_op1_dx20, L2_op1_dx20['L2'], 'r', label = r'UG100\_DX20',linewidth=4*FFIG)
plt.plot(t_L2_op1_dx20_no_turb, L2_op1_dx20_no_turb['L2'], '--r', label = r'UG100\_DX20\_NO\_TURB',linewidth=4*FFIG)
plt.plot(t_L2_op2_dx10, L2_op2_dx10['L2'], 'y', label = r'UG75\_DX10',linewidth=4*FFIG)
plt.plot(t_L2_op2_dx20, L2_op2_dx20['L2'], 'g', label = r'UG75\_DX20',linewidth=4*FFIG)
plt.xlabel(r'$\mathrm{t}^*$')
plt.ylabel(r'$L_2$')
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_expe_validation_L2_evolution.pdf')
#plt.savefig(folder_manuscript+'methods_expe_validation_L2_evolution.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# Error
# Trajectory difference
plt.figure(figsize=(FFIG*22,FFIG*15))
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot([-1,23],[0]*2,'k',linewidth=3*FFIG)
plt.plot(data_op1_dx10['xD'].values[1:], error_op1_dx10, 'b', label = r'UG100\_DX10',linewidth=4*FFIG)
plt.plot(data_op1_dx20['xD'].values[1:], error_op1_dx20, 'r', label = r'UG100\_DX20',linewidth=4*FFIG)
plt.plot(data_op1_dx20_no_turb['xD'].values[1:], error_op1_dx20_no_turb, '--r', label = r'UG100\_DX20\_NO\_TURB',linewidth=4*FFIG)
plt.plot(data_op2_dx10['xD'].values[1:], error_op2_dx10, 'y', label = r'UG75\_DX10',linewidth=4*FFIG)
plt.plot(data_op2_dx20['xD'].values[1:], error_op2_dx20, 'g', label = r'UG75\_DX20',linewidth=4*FFIG)
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\varepsilon~[\%]$')
plt.xlim(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
#plt.ylim(-5, 4)#becker_corr.plotD_limits[1][1])
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()




