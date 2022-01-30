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
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*30,FFIG*16) #(FFIG*22,FFIG*15)

##########################################################

x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')



z_max = 18

tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584



x_lim_traj = (-0.5, 20) #(becker_corr.plotD_limits[0][0], becker_corr.plotD_limits[0][1])
y_lim_traj = (becker_corr.plotD_limits[1][0], z_max)

dt = 1.5e-3 #ms

#%% Load and process trajectories  (method C)
method = 'method_c'


directory_op1_dx10 = folder+'/q6uG100/dx10'

L2_xD_01_02 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_01_02.csv')
L2_xD_03_04 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_03_04.csv')
L2_xD_05_06_07 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_05_06_07.csv')
L2_xD_08_09_10 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_08_09_10.csv')
L2_xD_11_12_13 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_11_12_13.csv')
L2_xD_14_15_16 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_14_15_16.csv')
L2_xD_17_18_19_20 = pd.read_csv(directory_op1_dx10+'/method_c_data_L2_all_xD_17_18_19_20.csv')


#%% Transform  L2 iterations to time
tp_0_UG100_DX10 = 1.9080932784636488
tp_max_UG100_DX10 = 3.5785496471047074 # diff of 0.5*tp

t = (L2_xD_01_02['t'].values - 1)*dt/tau_ph_UG100_DX10 + 2
m_op1_dx10 = (tp_max_UG100_DX10 - tp_0_UG100_DX10)/(t[-1] - t[0])
t = m_op1_dx10*(t - t[0]) + tp_0_UG100_DX10

#%% L2 error


plt.figure(figsize=figsize_)
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
plt.plot(t,L2_xD_01_02['L2_to_xD02'], 'b', label = r'$x/d_\mathrm{inj} < 2$')
plt.plot(t,L2_xD_05_06_07['L2_to_xD05'], 'r', label = r'$x/d_\mathrm{inj} < 5$')
plt.plot(t,L2_xD_08_09_10['L2_to_xD10'], 'k', label = r'$x/d_\mathrm{inj} < 10$')
plt.plot(t,L2_xD_14_15_16['L2_to_xD15'], 'g', label = r'$x/d_\mathrm{inj} < 15$')
plt.plot(t,L2_xD_01_02['L2'], '--k', label = r'$x/d_\mathrm{inj} < 22$')

#plt.xlabel(r'$\mathrm{t}^*$')
plt.xlabel(r'$t^{\prime}$')
plt.ylabel(r'$L_2$')
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1,
           bbox_to_anchor=(1.0, 1.0))
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
plt.savefig(folder_manuscript+'L2_evolution_with_xD_UG100_DX10.pdf')
plt.show()
plt.close()





