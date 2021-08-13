# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import pandas as pd

FFIG = 1.0
plt.rcParams['xtick.labelsize'] = 60*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 60*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True



figsize_ = (FFIG*26,FFIG*13)

#%% Cases

# Main folders
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_line_averages/'
folder = 'C:/Users/d601630/Desktop/Ongoing/ICS_study/TKE_evolution/cases_probes/'

# Cases
case_DX1p0 = folder + 'irene_mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5 = folder + 'irene_mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p3 = folder + 'irene_mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
         
cases = [case_DX1p0, case_DX0p5, case_DX0p3]


# Labels
labels_ = [r'$ \Delta x_\mathrm{ups} = 1.0 ~\mathrm{mm}$', r'$ \Delta x_\mathrm{ups} = 0.5 ~\mathrm{mm}$', r'$ \Delta x_\mathrm{ups} = 0.3 ~\mathrm{mm}$']



# axis labels
x_label_U_MEAN  = r"$t/\tau_\mathrm{fl}$"
y_label_U_MEAN  = r"$\langle u \rangle ~ [m . s^{-1}]$"
x_label_TKE = r"$t/\tau_\mathrm{fl}$"
y_label_TKE = r"$\langle TKE \rangle ~ [J . kg^{-1}]$"

# Format for lines
c1 = 'k'
c2 = 'b'
c3 = 'r'

# flow-through time [ms]
tau_ft = 1.2

#%% Read probes data



p_time_all_cases = []
p_U_all_cases = []
p_TKE_all_cases = []
p_TKE_w_RMS_all_cases = []
for i in cases:

    df = pd.read_csv(i+'data_line_average_probes.csv')
    
    p_time_all_cases.append(df['time'].values/tau_ft+6)
    p_U_all_cases.append(df['U_mean'].values)
    p_TKE_all_cases.append(df['TKE_mean'].values)
    p_TKE_w_RMS_all_cases.append(df['TKE_w_RMS_mean'].values)
    




#%% Plots



# U_mean
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases[0][1:], p_U_all_cases[0][1:], c1, label=labels_[0])
plt.plot(p_time_all_cases[1][1:], p_U_all_cases[1][1:], c2, label=labels_[1])
plt.plot(p_time_all_cases[2][1:], p_U_all_cases[2][1:], c3, label=labels_[2])
#plt.ylim(0, 10)
plt.xlabel(x_label_U_MEAN)
plt.ylabel(y_label_U_MEAN)
#plt.title('u signal')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'U_MEAN.pdf')
plt.savefig(folder_manuscript+'U_MEAN.eps',format='eps',dpi=1000)
plt.show()
plt.close()


# TKE
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases[0][1:], p_TKE_w_RMS_all_cases[0][1:], c1, label=labels_[0])
plt.plot(p_time_all_cases[1][1:], p_TKE_w_RMS_all_cases[1][1:], c2, label=labels_[1])
plt.plot(p_time_all_cases[2][1:], p_TKE_w_RMS_all_cases[2][1:], c3, label=labels_[2])
plt.ylim(0, 20)
plt.xlabel(x_label_TKE)
plt.ylabel(y_label_TKE)
#plt.title('u signal')
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'TKE.pdf')
plt.savefig(folder_manuscript+'TKE.eps',format='eps',dpi=1000)
plt.show()
plt.close()