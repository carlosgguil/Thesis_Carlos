# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:31:48 2020

@author: d601630
"""


import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/sprays_tools/jicf_penetration')
sys.path.append('C:/Users/Carlos Garcia/Desktop/Ongoing/JICF_trajectories/trajectories_SPS')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trajectory_calculation_functions as trj
from functions_methods import get_mean_trajectory_sweep



# Change size of figures if wished
FFIG = 0.5


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_trajectories/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF_trajectories/trajectories_SPS/data_trajectories'
# Physical parameters (lists, one value per figure)
q             = 6 #Becker 6, Ragucci 14.2  # Kinetic energy ratio [-]
d_inj         = 0.45  # Becker 0.45, Ragucci 0.5   # Injector diameter [mm]


# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*26,FFIG*16) #(FFIG*22,FFIG*15)

##########################################################

x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')


op1 = 'q6uG100'
op2 = 'q6uG75'

z_max = 18

tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584
tau_ph_UG100_DX20_NT = 0.2602





x_lim_traj = (-0.5, 20) #(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
y_lim_traj = (becker_corr.plotD_limits[1][0], z_max)

dt = 1.5e-3 #ms

tp_0_true_values = False
if tp_0_true_values:
    tp_0_UG100_DX20 = 0.6840/tau_ph_UG100_DX20
    tp_0_UG100_DX20_NT = 0.7844/tau_ph_UG100_DX20_NT
    tp_0_UG100_DX10 = 0.4173/tau_ph_UG100_DX10
    tp_0_UG75_DX20 = 0.9640/tau_ph_UG75_DX20
    tp_0_UG75_DX10 = 0.5032/tau_ph_UG75_DX10
else:
    tp_0_UG100_DX20 = 2.3
    tp_0_UG100_DX20_NT = 2.2
    tp_0_UG100_DX10 = 1.9080932784636488
    tp_0_UG75_DX20 = 2.3
    tp_0_UG75_DX10 = 1.85
    
    
# define maximum values for t' (obtained from ch5_nelem_plot.py)
tp_max_UG100_DX20 = 23.8370270548746 # diff of 1*tp
tp_max_UG100_DX20_NT = 23.436246263752494 # diff of 2*tp
tp_max_UG100_DX10 = 3.5785496471047074 # diff of 0.5*tp
tp_max_UG75_DX20 = 17.695510529612424 # no diff !
tp_max_UG75_DX10 = 3.6428757530101596 # no diff !

format_ = ['b','r','--k','y','g']

#%% Load and process trajectories  (method C)
method = 'method_c'

# OP1, dx = 20 µm
directory_op1_dx20 = folder+'/'+op1+'/dx20'
data_op1_dx20 = pd.read_csv(directory_op1_dx20+'/'+method+'_data_trajectory.csv')
L2_op1_dx20   = pd.read_csv(directory_op1_dx20+'/'+method+'_data_L2.csv')
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
t_L2_op1_dx20 = (L2_op1_dx20['t'].values - 1)*dt/tau_ph_UG100_DX20 + 2
t_L2_op1_dx20_no_turb = (L2_op1_dx20_no_turb['t'].values - 1)*dt/tau_ph_UG100_DX20_NT + 2
t_L2_op1_dx10 = (L2_op1_dx10['t'].values - 1)*dt/tau_ph_UG100_DX10 + 2

t_L2_op2_dx20 = (L2_op2_dx20['t'].values - 1)*dt/tau_ph_UG75_DX20 + 2
t_L2_op2_dx10 = (L2_op2_dx10['t'].values - 1)*dt/tau_ph_UG75_DX10 + 2

#%% 
# Now transform times (as needed)

# OP1_DX20
m_op1_dx20 = (tp_max_UG100_DX20 - tp_0_UG100_DX20)/(t_L2_op1_dx20[-1] - t_L2_op1_dx20[0])
t_L2_op1_dx20 = m_op1_dx20*(t_L2_op1_dx20 - t_L2_op1_dx20[0]) + tp_0_UG100_DX20

# OP1_DX20_no_turb
m_op1_dx20_NT = (tp_max_UG100_DX20_NT - tp_0_UG100_DX20_NT )/(t_L2_op1_dx20_no_turb[-1] - t_L2_op1_dx20_no_turb[0])
t_L2_op1_dx20_no_turb = m_op1_dx20_NT*(t_L2_op1_dx20_no_turb  - t_L2_op1_dx20_no_turb[0]) + tp_0_UG100_DX20_NT 

#OP1_DX10
m_op1_dx10 = (tp_max_UG100_DX10 - tp_0_UG100_DX10)/(t_L2_op1_dx10[-1] - t_L2_op1_dx10[0])
t_L2_op1_dx10 = m_op1_dx10*(t_L2_op1_dx10 - t_L2_op1_dx10[0]) + tp_0_UG100_DX10

# OP2_DX20
m_op2_dx20 = (tp_max_UG75_DX20 - tp_0_UG75_DX20)/(t_L2_op2_dx20[-1] - t_L2_op2_dx20[0])
t_L2_op2_dx20 = m_op2_dx20*(t_L2_op2_dx20 - t_L2_op2_dx20[0]) + tp_0_UG75_DX20

# OP2_DX10
m_op2_dx10 = (tp_max_UG75_DX10 - tp_0_UG75_DX10)/(t_L2_op2_dx10[-1] - t_L2_op2_dx10[0])
t_L2_op2_dx10 = m_op2_dx10*(t_L2_op2_dx10 - t_L2_op2_dx10[0]) + tp_0_UG75_DX10

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
plt.figure(figsize=figsize_)
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'$\mathrm{Exp.~correlation}$',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')

plt.plot(data_op1_dx10['xD'], data_op1_dx10['zD'], format_[0], label = r'$\mathrm{UG1}00\_\mathrm{DX}10$')
plt.plot(data_op1_dx20['xD'], data_op1_dx20['zD'], format_[1], label = r'$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(data_op1_dx20_no_turb['xD'], data_op1_dx20_no_turb['zD'], format_[2], label = r'$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$')
'''
plt.plot(data_op1_dx10['xD'], data_op1_dx10['zD'], 'b', label = r'$\Delta x_\Gamma = 10 \mu m$',linewidth=4*FFIG)
plt.plot(data_op1_dx20['xD'], data_op1_dx20['zD'], 'r', label = r'$\Delta x_\Gamma = 20 \mu m$',linewidth=4*FFIG)
plt.plot(data_op1_dx20_no_turb['xD'], data_op1_dx20_no_turb['zD'], '--r', label = r'$\Delta x_\Gamma = 20 \mu m,~\mathrm{no~turb.}$',linewidth=4*FFIG)
'''
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(x_lim_traj)
plt.ylim(y_lim_traj)
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
#plt.title(r'$(b)$ $q=6$, $We_\infty = 1470$')
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# OP2
plt.figure(figsize=figsize_)
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'$\mathrm{Exp.~correlation}$',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
'''
plt.plot(data_op2_dx10['xD'], data_op2_dx10['zD'], 'y', label = r'$\Delta x_\Gamma = 10 \mu m$',linewidth=4*FFIG)
plt.plot(data_op2_dx20['xD'], data_op2_dx20['zD'], 'g', label = r'$\Delta x_\Gamma = 20 \mu m$',linewidth=4*FFIG)
'''
plt.plot(data_op2_dx10['xD'], data_op2_dx10['zD'], format_[3], label = r'$\mathrm{UG75}\_\mathrm{DX}10$')
plt.plot(data_op2_dx20['xD'], data_op2_dx20['zD'], format_[4], label = r'$\mathrm{UG75}\_\mathrm{DX}20$')
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(x_lim_traj)
plt.ylim(y_lim_traj)
plt.grid()
plt.legend(numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
#plt.title(r'$(a)$ $q=6$, $We_\infty = 830$')
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG75.pdf')
#plt.savefig(folder_manuscript+'methods_expe_validation_trajectories_q6uG75.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% L2 error and error

linewidth_L2 = 8*FFIG

# L2 error
plt.figure(figsize=figsize_)
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)

plt.plot(t_L2_op1_dx10, L2_op1_dx10['L2_to_xD10'], format_[0], label = r'UG100\_DX10')
plt.plot(t_L2_op1_dx20, L2_op1_dx20['L2_to_xD10'], format_[1], label = r'UG100\_DX20')
plt.plot(t_L2_op1_dx20_no_turb, L2_op1_dx20_no_turb['L2_to_xD10'], format_[2], label = r'UG100\_DX20\_NO\_TURB')#,linewidth=10*FFIG)
plt.plot(t_L2_op2_dx10, L2_op2_dx10['L2_to_xD10'], format_[3], label = r'UG75\_DX10')
plt.plot(t_L2_op2_dx20, L2_op2_dx20['L2_to_xD10'], format_[4], label = r'UG75\_DX20')
'''
plt.plot(t_L2_op1_dx10, L2_op1_dx10['L2'], 'b', label = r'$\mathrm{UG}100\_\mathrm{DX}10$',linewidth=4*FFIG)
plt.plot(t_L2_op1_dx20, L2_op1_dx20['L2'], 'r', label = r'$\mathrm{UG}100\_\mathrm{DX}20$',linewidth=4*FFIG)
plt.plot(t_L2_op1_dx20_no_turb, L2_op1_dx20_no_turb['L2'], '--r', label = r'$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$',linewidth=4*FFIG)
plt.plot(t_L2_op2_dx10, L2_op2_dx10['L2'], 'y', label = r'$\mathrm{UG}75\_\mathrm{DX}10$',linewidth=4*FFIG)
plt.plot(t_L2_op2_dx20, L2_op2_dx20['L2'], 'g', label = r'$\mathrm{UG}75\_\mathrm{DX}20$',linewidth=4*FFIG)
'''
#plt.xlabel(r'$\mathrm{t}^*$')
plt.xlabel(r'$t^{\prime}$')
plt.ylabel(r'$L_2$')
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_L2_evolution.pdf')
#plt.savefig(folder_manuscript+'methods_expe_validation_L2_evolution.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# Error
# Trajectory difference
plt.figure(figsize=figsize_)
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot([-1,23],[0]*2,'k',linewidth=3*FFIG)
plt.plot(data_op1_dx10['xD'].values[1:], error_op1_dx10, format_[0], label = r'UG100\_DX10')
plt.plot(data_op1_dx20['xD'].values[1:], error_op1_dx20, format_[1], label = r'UG100\_DX20')
plt.plot(data_op1_dx20_no_turb['xD'].values[1:], error_op1_dx20_no_turb, format_[2], label = r'UG100\_DX20\_\mathrm{NT}')
plt.plot(data_op2_dx10['xD'].values[1:], error_op2_dx10, format_[3], label = r'UG75\_DX10')
plt.plot(data_op2_dx20['xD'].values[1:], error_op2_dx20, format_[4], label = r'UG75\_DX20')
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\varepsilon~[\%]$')
plt.xlim(x_lim_traj)
plt.ylim(-40, 42)#becker_corr.plotD_limits[1][1])
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_expe_validation_error_with_xD.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()




