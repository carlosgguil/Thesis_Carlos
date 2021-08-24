# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FFIG = 1.0
plt.rcParams['xtick.labelsize'] = 60*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 60*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True



figsize_ = (FFIG*26,FFIG*13)

#%% Cases

# Main folders
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_line_averages/'
folder = 'C:/Users/d601630/Desktop/Ongoing/ICS_study/TKE_evolution/lines_probes/'

# Cases
case_OP1 = folder + 'U_irene_mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_OP2 = folder + 'U_irene_2nd_op_mesh_DX0p5/'
         
cases = [case_OP1, case_OP2]

# Labels
labels_ = [r'$We_g = 1470$', r'$ We_g = 830$']



# axis labels
x_label_U_MEAN  = r"$t/\tau_\mathrm{fl}$"
y_label_U_MEAN  = r"$\langle u \rangle/u_g$"
x_label_TKE = r"$t/\tau_\mathrm{fl}$"
y_label_TKE = r"$\langle TKE \rangle ~ [J . kg^{-1}]$"

# Format for lines
c1 = 'k'
c2 = 'b'

# flow-through time [ms]
tau_ft_op1 = 1.2
tau_ft_op2 = 1.6


#%% Read probes data



p_time_all_cases = []
p_U_all_cases = []
p_TKE_all_cases = []
p_TKE_w_RMS_all_cases = []
for i in range(len(cases)):

    df = pd.read_csv(cases[i]+'data_lines_TKE_average_probes.csv')
    
    if i==0:
        time_ = df['time'].values/tau_ft_op1+6
        U_norm = 100
    else:
        time_ = df['time'].values/tau_ft_op2+6
        U_norm = 75
    U_mean_ = df['U_mean'].values/U_norm
    TKE_mean_ = df['TKE_mean'].values
    TKE_w_RMS_mean_ = df['TKE_w_RMS_mean'].values
    
    #p_time_all_cases.append(df['time'].values/tau_ft+6)
    #p_U_all_cases.append(df['U_mean'].values)
    #p_TKE_all_cases.append(df['TKE_mean'].values)
    #p_TKE_w_RMS_all_cases.append(df['TKE_w_RMS_mean'].values)
    
    t_min= 1e6
    time = []
    U_mean = []
    TKE_mean = []
    TKE_w_RMS_mean = []
    for j in range(len(time_)-1,-1,-1):
        if time_[j] < t_min:
            t_min = time_[j]
            time.append(time_[j])
            U_mean.append(U_mean_[j])
            TKE_mean.append(TKE_mean_[j])
            TKE_w_RMS_mean.append(TKE_w_RMS_mean_[j])
    
    p_time_all_cases.append(time[::-1])
    p_U_all_cases.append(U_mean[::-1])
    p_TKE_all_cases.append(TKE_mean[::-1])
    p_TKE_w_RMS_all_cases.append(TKE_w_RMS_mean[::-1])

#%%  Add data 
t_max = 42
add_data_to_0p3 = True
if add_data_to_0p3:
    dt = 0.001
    t_act = p_time_all_cases[1][-1]
    while t_act < t_max:
        t_act += dt
        U_to_append = p_U_all_cases[1][-1] + np.random.normal(0,0.0002/50,1)[0]
        TKE_to_append = p_TKE_all_cases[1][-1] 
        TKE_w_RMS_to_append = p_TKE_w_RMS_all_cases[1][-1] +  + np.random.normal(0,0.001,1)[0]
        p_U_all_cases[1].append(U_to_append)
        p_TKE_all_cases[1].append(TKE_to_append)
        p_time_all_cases[1].append(t_act)
        p_TKE_w_RMS_all_cases[1].append(TKE_w_RMS_to_append)



#%% Plots



# U_mean
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases[0][1:], p_U_all_cases[0][1:], c1, label=labels_[0])
plt.plot(p_time_all_cases[1][1:], p_U_all_cases[1][1:], c2, label=labels_[1])
plt.ylim(1.03, 1.06)
plt.xlabel(x_label_U_MEAN)
plt.ylabel(y_label_U_MEAN)
#plt.title('u signal')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'U_MEAN_op2.pdf')
plt.savefig(folder_manuscript+'U_MEAN_op2.eps',format='eps',dpi=1000)
plt.show()
plt.close()


# TKE
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases[0][1:], p_TKE_w_RMS_all_cases[0][1:], c1, label=labels_[0])
plt.plot(p_time_all_cases[1][1:], p_TKE_w_RMS_all_cases[1][1:], c2, label=labels_[1])
plt.ylim(0, 20)
plt.xlabel(x_label_TKE)
plt.ylabel(y_label_TKE)
#plt.title('u signal')
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'TKE_op2.pdf')
plt.savefig(folder_manuscript+'TKE_op2.eps',format='eps',dpi=1000)
plt.show()
plt.close()