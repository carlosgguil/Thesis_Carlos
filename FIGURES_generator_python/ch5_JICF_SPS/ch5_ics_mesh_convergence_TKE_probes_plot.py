# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FFIG = 0.5
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['lines.markersize'] =  30*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = False

figsize_ = (FFIG*20,FFIG*13)


#%% Cases


# Main folders
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_probes/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/ICS_study/frequential_analyses/cases_probes/'

# Cases
case_DX1p0 = folder + 'mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5 = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5_no_turb = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_no_turbulence/'
case_DX0p3 = folder + 'mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_OP2   = folder + '2nd_op_mesh_DX0p5/'
         
cases = [case_DX1p0, case_DX0p5, case_DX0p5_no_turb, case_DX0p3, case_OP2]




# Format for lines
c1 = 'k'
c2 = '--b'

# flow-through time [ms]
tau_ft = 1.2
tau_ft_op2 = 1.6

# x labels for up
t_threshold = 8
t_lim_min   = t_threshold
t_lim_max   = t_threshold + 1

# ylabels for up
y_ticks_up = [-4.0,-2.0,0.0,2.0,4.0]
y_lim_up   = [-5.0,5.0]

dx01p0_A = 3.4674439931589593E+000
dx00p5_A = 4.17 # 5.7805796217560523E+000 # real one
dx00p3_A = 4.2 # 6.6930899770650968E+000 # real one

dx01p0_B = 2.2875589511337373E+000
dx00p5_B = 4.14 # 4.1221873094628334E+000 # real one
dx00p3_B = 4.19 # 4.5171851927076591E+000 # real one


#%% Plot with values obtained from folder TKE_evolution
plt.rcParams['text.usetex'] = True


plt.figure(figsize=figsize_)


plt.plot(1, dx01p0_A, 'bo', label=r'$\mathrm{Probe}~\mathrm{A}$')
plt.plot(0.5, dx00p5_A, 'bo')
plt.plot(0.3, dx00p3_A, 'bo')


plt.plot(1,  dx01p0_B, 'ko',label=r'$\mathrm{Probe}~\mathrm{B}$')
plt.plot(0.5, dx00p5_B, 'ko')
plt.plot(0.3, dx00p3_B,  'ko')


plt.xlim(0.2,1.1)
plt.xlabel(r'$\Delta x_\mathrm{ups} ~[mm]$')
plt.ylabel(r'$TKE~[J.kg^{-1}]$')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'TKE_vs_dx_in_probes.pdf')
plt.show()
plt.close()


