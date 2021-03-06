# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 50*FFIG
plt.rcParams['ytick.labelsize'] = 50*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True
plt.rcParams['legend.framealpha'] = 1.0


figsize_ = (FFIG*26,FFIG*13)

#%% Cases

# Main folder
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_mean_profiles/'
folder = 'C:/Users/d601630/Desktop/Ongoing/ICS_study/u_mean_profiles/cases_probes/'

# Cases
case_DX1p0 = folder + 'mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5 = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p3 = folder + 'mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5_no_turb = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_no_turbulence/'
         
cases = [case_DX1p0, case_DX0p5, case_DX0p3, case_DX0p5_no_turb]


# Labels
labels_ = [r'$ \Delta x_\mathrm{ups} = 1.0 ~\mathrm{mm}$', r'$ \Delta x_\mathrm{ups} = 0.5 ~\mathrm{mm}$',
           r'$ \Delta x_\mathrm{ups} = 0.3 ~\mathrm{mm}$',r'$ \Delta x_\mathrm{ups} = 0.5 ~\mathrm{mm} ~\mathrm{no~turb.}$']



# axis labels
x_label_U_MEAN  = r"$\overline{u} ~ [m . s^{-1}]$"
x_label_TKE  = r"$TKE ~ [J . kg^{-1}]$"
y_label = r"$z~[mm]$"

# Format for lines
c1 = 'k'
c2 = 'b'
c3 = 'r'
c4 = '--b'

y_lim = (0,20)

#%% Read probes data



p_z_all_cases = []
p_U_MEAN_all_cases = []
p_TKE_all_cases = []
p_TKE_w_RMS_all_cases = []
for i in cases:

    df = pd.read_csv(i+'data_mean_profiles.csv')
    
    
    z_ = df['z'].values
    U_mean_ = df['U_mean'].values
    TKE_mean_ = df['TKE_mean'].values
    TKE_w_RMS_mean_ = df['TKE_w_RMS_mean'].values
    
    p_z_all_cases.append(z_)
    p_U_MEAN_all_cases.append(U_mean_)
    p_TKE_all_cases.append(TKE_mean_)
    p_TKE_w_RMS_all_cases.append(TKE_w_RMS_mean_)



#%% Plots



# U_mean
plt.figure(figsize=figsize_)
plt.plot(p_U_MEAN_all_cases[0], p_z_all_cases[0], c1, label=labels_[0])
plt.plot(p_U_MEAN_all_cases[1], p_z_all_cases[1], c2, label=labels_[1])
plt.plot(p_U_MEAN_all_cases[2], p_z_all_cases[2], c3, label=labels_[2])
plt.plot(p_U_MEAN_all_cases[3], p_z_all_cases[3], c4, label=labels_[3])
plt.ylim(0, 20)
plt.xlabel(x_label_U_MEAN)
plt.ylabel(y_label)
#plt.title('u signal')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'U_MEAN_profiles.pdf')
plt.savefig(folder_manuscript+'U_MEAN_profiles.eps',format='eps',dpi=1000)
plt.show()
plt.close()


# TKE
plt.figure(figsize=figsize_)
plt.plot(p_TKE_w_RMS_all_cases[0], p_z_all_cases[0], c1, label=labels_[0])
plt.plot(p_TKE_w_RMS_all_cases[1], p_z_all_cases[1], c2, label=labels_[1])
plt.plot(p_TKE_w_RMS_all_cases[2], p_z_all_cases[2], c3, label=labels_[2])
plt.plot(p_TKE_w_RMS_all_cases[3], p_z_all_cases[3], c4, label=labels_[3])
plt.xlim(0,90)
plt.ylim(0, 20)
plt.xlabel(x_label_TKE)
plt.ylabel(y_label)
#plt.title('u signal')
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'TKE_profiles.pdf')
plt.savefig(folder_manuscript+'TKE_profiles.eps',format='eps',dpi=1000)
plt.show()
plt.close()
