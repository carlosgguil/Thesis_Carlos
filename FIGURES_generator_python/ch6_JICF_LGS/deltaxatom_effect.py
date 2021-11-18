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
plt.rcParams['lines.markersize'] =  20*FFIG
#plt.rcParams['legend.loc']      = 'upper left'
plt.rcParams['text.usetex'] = True

##########################################################


SMD_expe = 31

# SMD at x = 80 mm
dx_atom = [0, 2, 4, 6, 10, 20]
SMD_x80 = [17.3, 18, 19, 20.5, 23.6, 29.75]

# SMD evolution with x
x = [10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80]
x_with_dx_atom = [10,20,30,40,50,60,70,80
                  ]
SMD_dx_atom_00mm = [91.66, 79.03, 54.24, 38.15, 29.07, 24.23, 21.55, 
                    19.97, 19.05, 18.48, 18.12, 17.53, 17.37,
                    17.23, 17.09, 17.01, 17.09]

SMD_dx_atom_10mm = [91.66, 91.66, 25.64, 24.19,                    
                    23.96, 23.81, 23.66, 23.66]

SMD_dx_atom_20mm = [91.66, 91.66, 91.66, 33.83,
                    30.32, 30.13, 29.94, 29.81]



#%% Plot SMD at x = 80 

plt.figure(figsize=(FFIG*22,FFIG*15))
plt.title(r'SMD at x = 80 mm')
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.plot((dx_atom[0],dx_atom[-1]),[SMD_expe]*2,'--k',label='Experiments')
plt.plot(dx_atom, SMD_x80, 'o-k',markersize=20*FFIG,label='Simulations')
plt.xlabel(r'$\Delta x_\mathrm{atom}~[\mathrm{mm}]$')
plt.ylabel(r'$SMD~[\mu\mathrm{m}]$')
#plt.xlim(x_lim_traj)
#plt.ylim(-40, 30)#becker_corr.plotD_limits[1][1])
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% Plot SMD evolution with x

plt.figure(figsize=(FFIG*30,FFIG*15))
plt.title(r'SMD evolution')
#plt.plot(becker_corr.xD, becker_corr.zD_mean, 'k', label=r'Exp. correlation',linewidth=8*FFIG)
#plt.fill_between(becker_corr.xD, becker_corr.zD_lower, 
#                 becker_corr.zD_upper, alpha=0.1, facecolor='black')
plt.scatter(80,SMD_expe,marker='*',s=2000*FFIG,label='Experiments')
plt.plot(x,SMD_dx_atom_00mm,'-ok',label=r'$\Delta x_\mathrm{atom}$ = 0 mm')
plt.plot(x_with_dx_atom,SMD_dx_atom_10mm,'-or',label=r'$\Delta x_\mathrm{atom}$ = 10 mm')
plt.plot(x_with_dx_atom,SMD_dx_atom_20mm,'-ob',label=r'$\Delta x_\mathrm{atom}$ = 20 mm')
plt.xlim(9,81)
plt.xlabel(r'$x$ [mm]')
plt.ylabel(r'$SMD~[\mu\mathrm{m}]$')
#plt.xlim(x_lim_traj)
#plt.ylim(-40, 30)#becker_corr.plotD_limits[1][1])
plt.xticks([10,20,30,40,50,60,70,80])
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()

