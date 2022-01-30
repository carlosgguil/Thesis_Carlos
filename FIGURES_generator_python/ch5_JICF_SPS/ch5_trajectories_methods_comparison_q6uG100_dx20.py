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

figsize_ = (FFIG*26,FFIG*19) #(FFIG*22,FFIG*15)
figsize_L2_share_y = (FFIG*30,FFIG*20)

##########################################################

x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')


z_max = 15.0

x_lim_traj = (-0.5, 17) #(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
x_lim_traj = (-0.5, 20) 
y_lim_traj = (becker_corr.plotD_limits[1][0], z_max)



tau = 0.2584 #0.3628
dt = 1.5e-3 #ms

tp_0 = 2.3
tp_max = 23.8370270548746 

label_L2_xD10 = r'$x/d_\mathrm{inj} < 10$'
label_L2_xD20 = r'$x/d_\mathrm{inj} < 20$'

#L2_x_ticks = [0,5,10,15,20]
L2_x_ticks = [2, 5, 10, 15, 20]

#%% Load and process trajectories 

# OP1, dx = 20 µm
directory_op1_dx20     = folder+'/q6uG100/dx20'
data_op1_dx20_method_a = pd.read_csv(directory_op1_dx20+'/method_a_data_trajectory.csv')
L2_op1_dx20_method_a   = pd.read_csv(directory_op1_dx20+'/method_a_data_L2.csv')
data_op1_dx20_method_b = pd.read_csv(directory_op1_dx20+'/method_b_data_trajectory.csv')
L2_op1_dx20_method_b   = pd.read_csv(directory_op1_dx20+'/method_b_data_L2.csv')
data_op1_dx20_method_c = pd.read_csv(directory_op1_dx20+'/method_c_data_trajectory.csv')
L2_op1_dx20_method_c   = pd.read_csv(directory_op1_dx20+'/method_c_data_L2.csv')
data_op1_dx20_method_d = pd.read_csv(directory_op1_dx20+'/method_d_data_trajectory.csv')
L2_op1_dx20_method_d   = pd.read_csv(directory_op1_dx20+'/method_d_data_L2.csv')


L2_method_a_xD20 = L2_op1_dx20_method_a['L2']
L2_method_a_xD10 = L2_op1_dx20_method_a['L2_to_xD10']
L2_method_b_xD20 = L2_op1_dx20_method_b['L2']
L2_method_b_xD10 = L2_op1_dx20_method_b['L2_to_xD10']
L2_method_c_xD20 = L2_op1_dx20_method_c['L2']
L2_method_c_xD10 = L2_op1_dx20_method_c['L2_to_xD10']
L2_method_d_xD20 = L2_op1_dx20_method_d['L2']
L2_method_d_xD10 = L2_op1_dx20_method_d['L2_to_xD10']

#%% Get errors with axial location
becker_corr_method_a = trj.trajectory_vertical(data_op1_dx20_method_a['xD'].values*d_inj, d_inj)
becker_corr_method_a.get_trajectory(d_inj, q, correlation='becker')
error_a = data_op1_dx20_method_a['zD'].values - becker_corr_method_a.zD_mean
error_a = error_a[1:]
error_a = error_a/becker_corr_method_a.zD_mean[1:]*100

becker_corr_method_b = trj.trajectory_vertical(data_op1_dx20_method_b['xD'].values*d_inj, d_inj)
becker_corr_method_b.get_trajectory(d_inj, q, correlation='becker')
error_b = data_op1_dx20_method_b['zD'].values - becker_corr_method_b.zD_mean
error_b = error_b[1:]
error_b = error_b/becker_corr_method_b.zD_mean[1:]*100

becker_corr_method_c = trj.trajectory_vertical(data_op1_dx20_method_c['xD'].values*d_inj, d_inj)
becker_corr_method_c.get_trajectory(d_inj, q, correlation='becker')
error_c = data_op1_dx20_method_c['zD'].values - becker_corr_method_c.zD_mean
error_c = error_c[1:]
error_c = error_c/becker_corr_method_c.zD_mean[1:]*100


becker_corr_method_d = trj.trajectory_vertical(data_op1_dx20_method_d['xD'].values*d_inj, d_inj)
becker_corr_method_d.get_trajectory(d_inj, q, correlation='becker')
error_d = data_op1_dx20_method_d['zD'].values - becker_corr_method_d.zD_mean
error_d = error_d[1:]
error_d = error_d/becker_corr_method_d.zD_mean[1:]*100

#%% OP1, dx = 20 µm all methods together
title = r'OP1'

# Trajectory
plt.figure(figsize=figsize_)
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'$\mathrm{Exp.~correlation}$',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op1_dx20_method_a['xD'], data_op1_dx20_method_a['zD'], 'r', label = r'$\mathrm{INST\_NM}$')
plt.plot(data_op1_dx20_method_b['xD'], data_op1_dx20_method_b['zD'], 'b', label = r'$\mathrm{INST\_M}$')
plt.plot(data_op1_dx20_method_c['xD'], data_op1_dx20_method_c['zD'], 'g', label = r'$\mathrm{MEAN\_GRAD}$')
plt.plot(data_op1_dx20_method_d['xD'], data_op1_dx20_method_d['zD'], 'y', label = r'$\mathrm{MEAN\_CONT}$')
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim(x_lim_traj)
plt.ylim(y_lim_traj)#becker_corr.plotD_limits[1][1])
plt.grid()
plt.minorticks_off()
plt.legend(loc='lower right', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_trajectories_q6uG100_dx20.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_trajectories_q6uG100_dx20.eps',format='eps',dpi=1000)
plt.show()
plt.close()


t_method_a = (L2_op1_dx20_method_a['t'].values - 1)*dt/tau + 2
t_method_b = (L2_op1_dx20_method_b['t'].values - 1)*dt/tau + 2
t_method_c = (L2_op1_dx20_method_c['t'].values - 1)*dt/tau + 2
t_method_d = (L2_op1_dx20_method_d['t'].values - 1)*dt/tau + 2

# Transform times
m_a = (tp_max - tp_0)/(t_method_a[-1] - t_method_a[0])
t_method_a = m_a*(t_method_a - t_method_a[0]) + tp_0

m_b = (tp_max - tp_0)/(t_method_b[-1] - t_method_b[0])
t_method_b = m_b*(t_method_b - t_method_b[0]) + tp_0

m_c = (tp_max - tp_0)/(t_method_c[-1] - t_method_c[0])
t_method_c = m_c*(t_method_c - t_method_c[0]) + tp_0

m_d = (tp_max - tp_0)/(t_method_d[-1] - t_method_d[0])
t_method_d = m_d*(t_method_d - t_method_d[0]) + tp_0

# L2 error
plt.figure(figsize=figsize_)
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)

plt.plot(t_method_a, L2_method_a_xD20, 'r', label = r'INST\_NM')
plt.plot(t_method_b, L2_method_b_xD20, 'b', label = r'INST\_M')
plt.plot(t_method_c, L2_method_c_xD20, 'g', label = r'MEAN\_GRAD')
plt.plot(t_method_d, L2_method_d_xD20, 'y', label = r'MEAN\_CONT')
'''
plt.plot(t_method_a, L2_method_a_xD10, 'r', label = r'INST\_NM')
plt.plot(t_method_b, L2_method_b_xD10, 'b', label = r'INST\_M')
plt.plot(t_method_c, L2_method_c_xD10, 'g', label = r'MEAN\_GRAD')
plt.plot(t_method_d, L2_method_d_xD10 'y', label = r'MEAN\_CONT')
'''
#plt.xlabel(r'$t^*$')
plt.xlabel(r'$t^{\prime}$')
plt.ylabel(r'$L_2$')
plt.grid()
plt.minorticks_off()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_L2_evolution_q6uG100_dx20.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_L2_evolution_q6uG100_dx20.eps',format='eps',dpi=1000)
plt.show()
plt.close()


 
# shared y axis
fig = plt.figure(figsize=figsize_L2_share_y)
gs = fig.add_gridspec(1, 2, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2) = gs.subplots(sharey='row')
# x/D < 10
ax1.plot(t_method_a, L2_method_a_xD10, 'r')
ax1.plot(t_method_b, L2_method_b_xD10, 'b')
ax1.plot(t_method_c, L2_method_c_xD10, 'g')
ax1.plot(t_method_d, L2_method_d_xD10, 'y')
ax1.set_title(label_L2_xD10)
# x/D < 20
ax2.plot(t_method_a, L2_method_a_xD20, 'r')
ax2.plot(t_method_b, L2_method_b_xD20, 'b')
ax2.plot(t_method_c, L2_method_c_xD20, 'g')
ax2.plot(t_method_d, L2_method_d_xD20, 'y')
ax2.set_title(label_L2_xD20)
axs.flat[0].set(ylabel = r'$L_2$')
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=r'$t^{\prime}$')
    ax.set_xlim((2,24.5))
    ax.xaxis.set_ticks(L2_x_ticks)
    ax.grid()
for ax in axs.flat[1:]:
    ax.spines['left'].set_linewidth(6*FFIG)
    ax.spines['left'].set_linestyle('-.')
#plt.ylabel([0,2,4,6,8, 10])
#plt.ylim(0,10)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_L2_evolution_q6uG100_dx20_shared_y_axis.pdf')
plt.show()
plt.close

#%%

# Trajectory difference
plt.figure(figsize=figsize_)
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot([-1,23],[0]*2,'k',linewidth=3*FFIG)
plt.plot(data_op1_dx20_method_a['xD'].values[1:], error_a, 'r', label = r'$\mathrm{INST\_NM}$')
plt.plot(data_op1_dx20_method_b['xD'].values[1:], error_b, 'b', label = r'$\mathrm{INST\_M}$')
plt.plot(data_op1_dx20_method_c['xD'].values[1:], error_c, 'g', label = r'$\mathrm{MEAN\_GRAD}$')
plt.plot(data_op1_dx20_method_d['xD'].values[1:], error_d, 'y', label = r'$\mathrm{MEAN\_CONT}$')
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\varepsilon~[\%]$')
plt.xlim(x_lim_traj)
plt.ylim(-50, 10)#becker_corr.plotD_limits[1][1])
plt.yticks(np.arange(-50, 10, 10))
plt.grid()
plt.minorticks_off()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100_dx20.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100_dx20.eps',format='eps',dpi=1000)
plt.show()
plt.close()




