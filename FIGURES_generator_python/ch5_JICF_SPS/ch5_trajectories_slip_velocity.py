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


#labels_ = [r'UG100\_DX10', r'UG100\_DX20', r'UG100\_DX20\_NO\_TURB', r'UG75\_DX10', r'UG75\_DX20']
labels_ = [r'UG100\_DX10', r'UG100\_DX20', r'UG75\_DX10', r'UG75\_DX20']

# X coordinates: [5, 6, 7, 8, 9, 10]
# Z coord. UG100_DX10: [5.69, 6.2, 6.52, 7.52, 8.07, 7.77]
# Z coord. UG100_DX20: [3.83, 4, 4.1, 3.92, 3.99, 4.62]
# Z coord. UG100_DX20_NO_TURB: [3.83, 4, 4.1, 4.18, 4.37, 4.62]
# Z coord. UG75_DX10: [5.6, 6.14, 6.73, 6.94, 7.52, 7.59]
# Z coord. UG75_DX20: [3.9, 4.77, 4.31, 4.55, 4.66, 4.46]

x  = [5, 6, 7, 8, 9, 10]
xD = np.array(x)/d_inj

ug_mean_x_qUG100_DX10 = [64.64, 66.19, 62.34, 56.03, 57.71, 68.21]
ug_mean_z_qUG100_DX10 = [3.65, 2.34, 7.18, 6.96, 6.03, 6.54]
ul_mean_x_qUG100_DX10 = [68.71, 66.75, 69.65, 71.2, 70.62, 71.83]
ul_mean_z_qUG100_DX10 = [41.48, 51.32, 45.48, 50.3, 53.82, 41.12]
u_slip_x_qUG100_DX10 = np.array(ul_mean_x_qUG100_DX10) - np.array(ug_mean_x_qUG100_DX10)
u_slip_z_qUG100_DX10 = np.array(ul_mean_z_qUG100_DX10) - np.array(ug_mean_z_qUG100_DX10)

ug_mean_x_qUG100_DX20 = [72.01, 71.8, 69.3, 60.09, 61.54, 75.87]
ug_mean_z_qUG100_DX20 = [2.39, 2.71, 2.19, 2.15, 2.69, 2.77]
ul_mean_x_qUG100_DX20 = [52.59, 54, 57.41, 56.11, 56.32, 59.27]
ul_mean_z_qUG100_DX20 = [6.41, 7.19, 9.65, 6.26, 6.23, 9.36]
u_slip_x_qUG100_DX20 = np.array(ul_mean_x_qUG100_DX20) - np.array(ug_mean_x_qUG100_DX20)
u_slip_z_qUG100_DX20 = np.array(ul_mean_z_qUG100_DX20) - np.array(ug_mean_z_qUG100_DX20)


ug_mean_x_qUG75_DX10 = [50.06, 51.55, 47.89, 42.19, 40.94, 48.1 ]
ug_mean_z_qUG75_DX10 = [3.99, 12.37, 6.27, 5.41, 6.49, 7.41 ]
ul_mean_x_qUG75_DX10 = [52.6,  52.2,  53.45, 54.71, 53.08, 53.66]
ul_mean_z_qUG75_DX10 = [36.77, 40.63, 33.34, 35.68, 27.22, 26.58]
u_slip_x_qUG75_DX10 = np.array(ul_mean_x_qUG75_DX10) - np.array(ug_mean_x_qUG75_DX10)
u_slip_z_qUG75_DX10 = np.array(ul_mean_z_qUG75_DX10) - np.array(ug_mean_z_qUG75_DX10)

ug_mean_x_qUG75_DX20 = [59.14, 73.9, 61.03, 63.75, 62.99, 53.74]
ug_mean_z_qUG75_DX20 = [2.74, 3.06, 3.32, 3.15, 3.11, 2.57]
ul_mean_x_qUG75_DX20 = [37.78, 45.97, 44.19, 42.67, 47.56, 46.16]
ul_mean_z_qUG75_DX20 = [4.41, 12.92, 8.14, 4.93, 13.52, 8.44]
u_slip_x_qUG75_DX20 = np.array(ul_mean_x_qUG75_DX20) - np.array(ug_mean_x_qUG75_DX20)
u_slip_z_qUG75_DX20 = np.array(ul_mean_z_qUG75_DX20) - np.array(ug_mean_z_qUG75_DX20)


#%% 
# plot slip velocity in x
plt.figure(figsize=(FFIG*30,FFIG*15))
plt.title(r'Mean slip velocity in x')
plt.plot(xD, u_slip_x_qUG100_DX10, 'k', label=labels_[0])
plt.plot(xD, u_slip_x_qUG100_DX20, '--k', label=labels_[1])
plt.plot(xD, u_slip_x_qUG75_DX10, 'b', label=labels_[2])
plt.plot(xD, u_slip_x_qUG75_DX20, '--b', label=labels_[3])
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{u}_{\mathrm{sl},x}~[m/s]$')
plt.ylim(-30,20)
plt.grid()
plt.legend(numpoints = 2, framealpha=1, bbox_to_anchor=(1.0, 1.0))
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()


# plot slip velocity in z
plt.figure(figsize=(FFIG*30,FFIG*15))
plt.title(r'Mean slip velocity in z')
plt.plot(xD, u_slip_z_qUG100_DX10, 'k', label=labels_[0])
plt.plot(xD, u_slip_z_qUG100_DX20, '--k', label=labels_[1])
plt.plot(xD, u_slip_z_qUG75_DX10, 'b', label=labels_[2])
plt.plot(xD, u_slip_z_qUG75_DX20, '--b', label=labels_[3])
plt.xlabel(r'$x/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{u}_{\mathrm{sl},z}~[m/s]$')
plt.grid()
plt.legend(numpoints = 2, framealpha=1, bbox_to_anchor=(1.0, 1.0))
#plt.title(r' $u_G = 100$ m/s, $\Delta x_\mathrm{min}$ = 20 $\mu$m')
#plt.title(title)
plt.tight_layout()
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.pdf')
#plt.savefig(folder_manuscript+'methods_comparison_error_with_xD_q6uG100.eps',format='eps',dpi=1000)
plt.show()
plt.close()




