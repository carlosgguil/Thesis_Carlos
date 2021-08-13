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
plt.rcParams['text.usetex'] = False


figsize_ = (FFIG*26,FFIG*13)

#%% Cases

# Main folder
folder_case = './cases_probes/'
cases = [folder_case + 'irene_mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/',
         folder_case + 'irene_mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/',
         folder_case + 'irene_mesh_refined_DX0p5_ics_no_actuator_flat_BL_no_turbulence/',
         folder_case + 'irene_mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/']
         
cases = [folder_case + 'irene_mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/']

#%% Read probes

time_all_cases = []
z_all_cases = []
U_all_cases = []
TKE_all_cases = []
TKE_w_RMS_all_cases = []
for k in cases:
    
    # Read probes
    probe_U = pd.read_csv(k+'line_TKE_U_MEAN.dat',sep='(?<!\\#)\s+',engine='python')      
    probe_U.columns =  [name.split(':')[1] for name in probe_U.columns]
    probe_TKE = pd.read_csv(k+'line_TKE_TKE.dat',sep='(?<!\\#)\s+',engine='python')      
    probe_TKE.columns =  [name.split(':')[1] for name in probe_TKE.columns]
    probe_TKE_w_RMS = pd.read_csv(k+'line_TKE_TKE_w_RMS.dat',sep='(?<!\\#)\s+',engine='python')      
    probe_TKE_w_RMS.columns =  [name.split(':')[1] for name in probe_TKE_w_RMS.columns]
    time = probe_U['total_time'].values
 
    time             = []
    z_t              = []
    U_mean_t         = []
    TKE_mean_t       = []
    TKE_w_RMS_mean_t = []
    
    z_i = []
    U_i = []
    TKE_i = []
    TKE_w_RMS_i = []
    for i in range(len(probe_TKE)):
        
        # Check if we change time instant or not
        if not np.isnan(probe_U.loc[i]['total_time']):
            z_i.append(probe_U.loc[i]['Z']*1e3)
            U_i.append(probe_U.loc[i]['U_MEAN(1)'])
            TKE_i.append(probe_TKE.loc[i]['TKE'])
            TKE_w_RMS_i.append(probe_TKE_w_RMS.loc[i]['TKE_w_RMS'])
            
        else:
            
            time.append(probe_U.loc[i-1]['total_time']*1e3)
            z_t.append(z_i)
            U_mean_t.append(U_i)
            TKE_mean_t.append(TKE_i)
            TKE_w_RMS_mean_t.append(TKE_w_RMS_i)
            
            
            z_i = []
            U_i = []
            TKE_i = []
            TKE_w_RMS_i = []
    
    time_all_cases.append(time)
    z_all_cases.append(z_t)
    U_all_cases.append(U_mean_t)
    TKE_all_cases.append(TKE_mean_t)
    TKE_w_RMS_all_cases.append(TKE_w_RMS_mean_t)
    


    
# Filter repeated time values
p_time_all_cases = []
p_z_all_cases = []
p_u_all_cases = []
p_TKE_all_cases = []
p_TKE_w_RMS_all_cases = []
for i in range(len(cases)):
    time = time_all_cases[i]
    z    = z_all_cases[i]
    u    = U_all_cases[i]
    TKE  = TKE_all_cases[i]
    TKE_w_RMS  = TKE_w_RMS_all_cases[i]
    
    p_time  = []; time_max = -1
    p_u     = []; p_z = []
    p_TKE   = []; p_TKE_w_RMS = []
    for j in range(len(time)):
        t = time[j]
        if t > time_max:
            p_time.append(t)
            p_z.append(z[j])
            p_u.append(u[j])
            p_TKE.append(TKE[j])
            p_TKE_w_RMS.append(TKE_w_RMS[j])
            time_max = t

    
    p_time_all_cases.append(np.array(p_time))
    p_z_all_cases.append(np.array(p_z))
    p_u_all_cases.append(np.array(p_u))
    p_TKE_all_cases.append(np.array(p_TKE))
    p_TKE_w_RMS_all_cases.append(np.array(p_TKE_w_RMS))



#%% Write data to .csv
    
for i in range(len(cases)):
    z = p_z_all_cases[i][-1]
    U = p_u_all_cases[i][-1]
    TKE = p_TKE_all_cases[i][-1]
    TKE_w_RMS = p_TKE_w_RMS_all_cases[i][-1]
    d = {'z': z, 'U_mean': U, 'TKE_mean': TKE, 'TKE_w_RMS_mean': TKE_w_RMS}
    df = pd.DataFrame(data = d)
    df.to_csv(cases[i]+'data_mean_profiles.csv', index = False)


#%% 
    
    

labels_ = cases
#labels_ = ['Case 1 (no turb.)', 'Case 3 (with turb.)'] 
    
# Plot U_MEAN profile
plt.figure(figsize=figsize_)
for i in range(len(cases)):
    for j in range(len(p_u_all_cases[i])):
        plt.plot(p_u_all_cases[i][j], p_z_all_cases[i][j])
#plt.ylim(-13, 13)
plt.xlabel(r'u [m/s]')
plt.ylabel(r"$z ~[\mathrm{mm}]$")
plt.title("U_MEAN profile")
#plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()

# Plot TKE profile
plt.figure(figsize=figsize_)
for i in range(len(cases)):
    for j in range(len(p_u_all_cases[i])):
        plt.plot(p_TKE_all_cases[i][j], p_z_all_cases[i][j])
#plt.ylim(-13, 13)
plt.xlabel(r'TKE [J/kg]')
plt.ylabel(r"$z ~[\mathrm{mm}]$")
plt.title("TKE profile")
#plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()


# Plot TKE_w_RMS profile
plt.figure(figsize=figsize_)
for i in range(len(cases)):
    for j in range(len(p_u_all_cases[i])):
        plt.plot(p_TKE_w_RMS_all_cases[i][j], p_z_all_cases[i][j])
#plt.ylim(-13, 13)
plt.xlabel(r'TKE [J/kg]')
plt.ylabel(r"$z ~[\mathrm{mm}]$")
plt.title("TKE_w_RMS profile")
#plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()

