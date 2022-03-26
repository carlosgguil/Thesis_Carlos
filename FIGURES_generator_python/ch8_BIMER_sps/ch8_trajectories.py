# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:31:48 2020

@author: d601630
"""


import sys
sys.path.append('C:/Users/Carlos Garcia/Desktop/Ongoing/JICF_trajectories/trajectories_SPS')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trajectory_calculation_functions as trj
from functions_methods import get_mean_trajectory_sweep



# Change size of figures if wished
FFIG = 0.5


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/trajectories/'


# Physical parameters (BIMER)
q             = 2 #Becker 6, Ragucci 14.2  # Kinetic energy ratio [-]
d_inj         = 0.3  # Becker 0.45, Ragucci 0.5   # Injector diameter [mm]


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

x_corr          = np.linspace(0,2,1000) #np.linspace(0, 22*D, 50)
becker_corr = trj.trajectory_vertical(x_corr, d_inj)
becker_corr.get_trajectory(d_inj, 2, correlation='becker')
becker_corr.zD_upper[0] = 0









# DX10
file_DX10 = folder+'/dx10/BIMER_trajectory.csv'
data_DX10 = pd.read_csv(file_DX10)


# DX15
file_DX15 = folder+'/dx15/BIMER_trajectory.csv'
data_DX15 = pd.read_csv(file_DX15)




#%% Trajectories 


plt.figure(figsize=figsize_)
plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'$\mathrm{Exp.~correlation}$',linewidth=8*FFIG)
plt.plot(data_DX15['xD'], data_DX15['zD'],'b', label=r'$\mathrm{DX15}$',linewidth=8*FFIG)
plt.plot(data_DX10['xD'], data_DX10['zD'],'r', label=r'$\mathrm{DX10}$',linewidth=8*FFIG)
plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
                 becker_corr.zD_upper, alpha=0.1, facecolor='black')

#plt.plot(data_op1_dx10['xD'], data_op1_dx10['zD'], format_[0], label = r'$\mathrm{UG1}00\_\mathrm{DX}10$')
plt.xlabel(r'$x_c/d_\mathrm{inj}$')
plt.ylabel(r'$z_c/d_\mathrm{inj}$')
plt.xlim((-0.15, 6.5) )
plt.xticks([0,1,2,3,4,5,6])
#plt.yticks([0,1,2,3,4,5,6,7,8])
plt.ylim((0, 8))
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
#plt.title(r'$(b)$ $q=6$, $We_\infty = 1470$')
plt.tight_layout()
plt.savefig(folder_manuscript+'trajectories_BIMER.pdf')
plt.show()
plt.close()





