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

#%% Read probes
time_all_cases_line_0 = []
u_all_cases_line_0 = []
xf_all_cases_line_0 = []
y_FFT_all_cases_line_0 = []


time_all_cases_line_inj = []
u_all_cases_line_inj = []
xf_all_cases_line_inj = []
y_FFT_all_cases_line_inj = []

TKE_line_0 = []
TKE_line_2 = []
TKE_line_inj = []
for i in range(len(cases)):
    case = cases[i]
    
    df_line_0_up1       = pd.read_csv(case+'data_line_0_up1.csv')
    df_line_0_up2       = pd.read_csv(case+'data_line_0_up2.csv')
    df_line_0_up3       = pd.read_csv(case+'data_line_0_up3.csv')
    df_line_2_up1       = pd.read_csv(case+'data_line_2_up1.csv')
    df_line_2_up2       = pd.read_csv(case+'data_line_2_up2.csv')
    df_line_2_up3       = pd.read_csv(case+'data_line_2_up3.csv')
    df_line_inj_up1      = pd.read_csv(case+'data_line_inj_up1.csv')
    df_line_inj_up2      = pd.read_csv(case+'data_line_inj_up2.csv')
    df_line_inj_up3      = pd.read_csv(case+'data_line_inj_up3.csv')
    
    t = df_line_0_up1['t'].values
    T = t[-1]-t[0]
    
    line_0_up1_sq = df_line_0_up1['up'].values**2
    line_0_up2_sq = df_line_0_up2['up'].values**2
    line_0_up3_sq = df_line_0_up3['up'].values**2
    
    
    line_2_up1_sq = df_line_2_up1['up'].values**2
    line_2_up2_sq = df_line_2_up2['up'].values**2
    line_2_up3_sq = df_line_2_up3['up'].values**2
    
    line_inj_up1_sq = df_line_inj_up1['up'].values**2
    line_inj_up2_sq = df_line_inj_up2['up'].values**2
    line_inj_up3_sq = df_line_inj_up3['up'].values**2
    
    line0_k1 = 0; line2_k1 = 0; lineinj_k1 = 0
    line0_k2 = 0; line2_k2 = 0; lineinj_k2 = 0
    line0_k3 = 0; line2_k3 = 0; lineinj_k3 = 0
    for j in range(len(t)-1):
        dt = t[j+1] - t[j]
        
        line0_k1 += 0.5*dt*(line_0_up1_sq[j+1] + line_0_up1_sq[j])
        line0_k2 += 0.5*dt*(line_0_up2_sq[j+1] + line_0_up2_sq[j])
        line0_k3 += 0.5*dt*(line_0_up3_sq[j+1] + line_0_up1_sq[j])
        
        line2_k1 += 0.5*dt*(line_0_up1_sq[j+1] + line_0_up1_sq[j])
        line2_k2 += 0.5*dt*(line_2_up2_sq[j+1] + line_2_up2_sq[j])
        line2_k3 += 0.5*dt*(line_2_up3_sq[j+1] + line_2_up1_sq[j])
        
        
        lineinj_k1 += 0.5*dt*(line_inj_up1_sq[j+1] + line_inj_up1_sq[j])
        lineinj_k2 += 0.5*dt*(line_inj_up2_sq[j+1] + line_inj_up2_sq[j])
        lineinj_k3 += 0.5*dt*(line_inj_up3_sq[j+1] + line_inj_up1_sq[j])
    
    line0_k1 = line0_k1/T; line2_k1 = line2_k1/T; lineinj_k1 = lineinj_k1/T
    line0_k2 = line0_k2/T; line2_k2 = line2_k2/T; lineinj_k2 = lineinj_k2/T
    line0_k3 = line0_k3/T; line2_k3 = line2_k3/T; lineinj_k3 = lineinj_k3/T
    
    line0_TKE = 0.5*(line0_k1+line0_k2+line0_k3)
    line2_TKE = 0.5*(line2_k1+line2_k2+line2_k3)
    lineinj_TKE = 0.5*(lineinj_k1+lineinj_k2+lineinj_k3)
    
    TKE_line_0.append(line0_TKE)
    TKE_line_2.append(line2_TKE)
    TKE_line_inj.append(lineinj_TKE)
    
    
#%% Plot

plt.figure(figsize=figsize_)
plt.plot(1, TKE_line_0[0], 'ro', label='Line 0')
plt.plot(0.5, TKE_line_0[1], 'ro')
plt.plot(0.3, TKE_line_0[3], 'ro')
plt.plot(1, TKE_line_2[0], 'bo', label='Line 2')
plt.plot(0.5, TKE_line_2[1], 'bo')
plt.plot(0.3, TKE_line_2[3], 'bo')
plt.plot(1, TKE_line_inj[0], 'ko', label='Line inj')
plt.plot(0.5, TKE_line_inj[1], 'ko')
plt.plot(0.3, TKE_line_inj[3], 'ko')
'''
plt.plot(0.5, TKE_line_0[2], 'r*', label='No turb.')
plt.plot(0.5, TKE_line_2[2], 'b*')
plt.plot(0.5, TKE_line_inj[2], 'k*')
'''
plt.xlim(0.2,1.1)
plt.xlabel('$\Delta x_\mathrm{ups} ~[mm]$')
plt.ylabel('$TKE$ ~[J.kg^{-1}]')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.show()
plt.close()

#%% Plot with values obtained from folder TKE_evolution
plt.rcParams['text.usetex'] = True


plt.figure(figsize=figsize_)
'''
plt.plot(1, TKE_line_0[0], 'ro', label='Line 0')
plt.plot(0.5, TKE_line_0[1], 'ro')
plt.plot(0.3, TKE_line_0[3], 'ro')
'''
plt.plot(1, TKE_line_2[0], 'bo', label=r'$\mathrm{Probe}~\mathrm{A}$')
plt.plot(0.5, TKE_line_2[1], 'bo')
plt.plot(0.3, TKE_line_2[3], 'bo')


plt.plot(1, 2.29049, 'ko',label=r'$\mathrm{Probe}~\mathrm{B}$')
plt.plot(0.5, 4.12045, 'ko')
plt.plot(0.3, 4.52054, 'ko')

plt.xlim(0.2,1.1)
plt.xlabel(r'$\Delta x_\mathrm{ups} ~[mm]$')
plt.ylabel(r'$TKE~[J.kg^{-1}]$')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'TKE_vs_dx_in_probes.pdf')
plt.show()
plt.close()


plt.rcParams['text.usetex'] = False

