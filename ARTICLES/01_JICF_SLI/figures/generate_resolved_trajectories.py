# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:31:48 2020

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trajectory_calculation_functions as trj



# Change size of figures if wished
FFIG = 0.5


# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True




#%%

# Physical parameters =
q             = 6 
d_inj         = 0.45  

# experimental correlqtion
x_corr          = np.linspace(0,10,50) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, q, correlation='becker')



#%% Load and process trajectories  (method C)

data_op1_dx20 = pd.read_csv('./data_resolved_trajectories/q6uG100_dx20.csv')
#data_op1_dx20_no_turb =  pd.read_csv('./data_resolved_trajectories/q6uG100_dx20_NT.csv')
data_op1_dx10 = pd.read_csv('./data_resolved_trajectories/q6uG100_dx10.csv')


data_op2_dx20 = pd.read_csv('./data_resolved_trajectories/q6uG75_dx20.csv')
data_op2_dx10 = pd.read_csv('./data_resolved_trajectories/q6uG75_dx10.csv')


#%% Trajectories 

# OP1
plt.figure(figsize=(FFIG*26,FFIG*18))
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'$\mathrm{Exp.~correlation}$',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot(data_op1_dx10['xD'], data_op1_dx10['zD'], 'b', label = r'$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(data_op1_dx20['xD'], data_op1_dx20['zD'], 'r', label = r'$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(data_op2_dx10['xD'], data_op2_dx10['zD'], '--b', label = r'$\mathrm{UG}75\_\mathrm{DX}10$')
plt.plot(data_op2_dx20['xD'], data_op2_dx20['zD'], '--r', label = r'$\mathrm{UG}75\_\mathrm{DX}20$')
#plt.plot(data_op1_dx20_no_turb['xD'], data_op1_dx20_no_turb['zD'], format_[2], label = r'$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$')
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$z/d_\mathrm{inj}$')
plt.xlim((-0.5,20))
plt.ylim((becker_corr.plotD_limits[1][0], 18))
plt.grid()
plt.legend(loc='lower right', ncol=2, numpoints = 2, framealpha=1)
plt.tight_layout()
plt.savefig('./resolved_trajectories.pdf')
plt.show()
plt.close()
