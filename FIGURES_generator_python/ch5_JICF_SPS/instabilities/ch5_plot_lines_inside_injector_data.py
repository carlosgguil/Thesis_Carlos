# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:17:29 2022

@author: Carlos Garcia
"""

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['lines.markersize'] =  40*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*20,FFIG*16)


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/instabilities_resolution/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/instabilities_injector_observation/data_lines_inside_injector/'


label_x  = '$x/r_\mathrm{inj}$'#'$x ~[\mathrm{mm}]$'
#label_u  = r'$\overline{u_z} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_u  = r'$\overline{w} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_vort  = r'$|\overline{\omega}| (\times 10^5) ~[\mathrm{s}^{-1}$]'
label_TKE  = r'$TKE ~[\mathrm{J}~\mathrm{kg}^{-1}$]'
label_dx10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_dx20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_closer = r'$z_\mathrm{low}$' #r'$\mathrm{Closer~injector}$'
label_further = r'$z_\mathrm{up}$' #r'$\mathrm{Further~injector}$'


r_inj = 0.45/2
x_lim_ = (-1,0) #(-0.225,0)
y_lim_uz = (0,30)

labels_z_lines = [r'$z = 0~\mathrm{mm}$',
                  r'$z = -0.35~\mathrm{mm}$']
labels_save_figs = ['z0p00','zm0p35']


df_dx10_zm0p00mm = pd.read_csv(folder+'UG100_DX10_line_inside_injector_zm0p00mm.csv')
df_dx10_zm0p12mm = pd.read_csv(folder+'UG100_DX10_line_inside_injector_zm0p12mm.csv')
df_dx10_zm0p35mm = pd.read_csv(folder+'UG100_DX10_line_inside_injector_zm0p35mm.csv')
df_dx10_zm0p70mm = pd.read_csv(folder+'UG100_DX10_line_inside_injector_zm0p70mm.csv')

df_dx20_zm0p00mm = pd.read_csv(folder+'UG100_DX20_line_inside_injector_zm0p00mm.csv')
df_dx20_zm0p12mm = pd.read_csv(folder+'UG100_DX20_line_inside_injector_zm0p12mm.csv')
df_dx20_zm0p35mm = pd.read_csv(folder+'UG100_DX20_line_inside_injector_zm0p35mm.csv')
df_dx20_zm0p70mm = pd.read_csv(folder+'UG100_DX20_line_inside_injector_zm0p70mm.csv')


dfs_dx10 = [df_dx10_zm0p00mm, df_dx10_zm0p35mm]
dfs_dx20 = [df_dx20_zm0p00mm, df_dx20_zm0p35mm]

#%%

# get data dx10
x_dx10 = []; p_dx10 = []; p_mean_dx10 = []
ux_dx10 = []; uy_dx10 = []; uz_dx10 = []
ux_mean_dx10 = []; uy_mean_dx10 = []; uz_mean_dx10 = []; 
ux_RMS_dx10 = []; uy_RMS_dx10 = []; uz_RMS_dx10 = []; 
vort_x_dx10 = []; vort_y_dx10 = []; vort_z_dx10 = []; 
vort_x_mean_dx10 = []; vort_y_mean_dx10 = []; vort_z_mean_dx10 = [];
TKE_dx10 = []; vort_mag_dx10 = []; vort_mean_mag_dx10 =  []
for i in range(len(dfs_dx10)):
    df = dfs_dx10[i]
    
    x_i    = df['Points_0'].values*1e3/r_inj
    p_i    = df['P'].values*1e3
    p_mean_i = df['P_MEAN'].values*1e3
    ux_i   = df['U_0'].values
    uy_i   = df['U_1'].values
    uz_i   = df['U_2'].values
    ux_mean_i   = df['U_MEAN_0'].values
    uy_mean_i   = df['U_MEAN_1'].values
    uz_mean_i   = df['U_MEAN_2'].values
    ux_RMS_i   = df['U_RMS_0'].values
    uy_RMS_i   = df['U_RMS_1'].values
    uz_RMS_i   = df['U_RMS_2'].values
    vort_x_i = df['VORT_0'].values
    vort_y_i = df['VORT_1'].values
    vort_z_i = df['VORT_2'].values
    vort_x_mean_i = df['VORT_MEAN_0'].values
    vort_y_mean_i = df['VORT_MEAN_1'].values
    vort_z_mean_i = df['VORT_MEAN_2'].values
    
    TKE_i = 0.5*(ux_RMS_i**2+uy_RMS_i**2+uz_RMS_i**2)
    vort_mag_i = np.sqrt(vort_x_i**2+vort_y_i**2+vort_z_i**2)
    vort_mean_mag_i = np.sqrt(vort_x_mean_i**2+vort_y_mean_i**2+vort_z_mean_i**2)
    
    
    x_dx10.append(x_i)
    p_dx10.append(p_i)
    p_mean_dx10.append(p_mean_i)
    
    ux_dx10.append(ux_i)
    uy_dx10.append(uy_i)
    uz_dx10.append(uz_i)
    ux_mean_dx10.append(ux_mean_i)
    uy_mean_dx10.append(uy_mean_i)
    uz_mean_dx10.append(uz_mean_i)
    ux_RMS_dx10.append(ux_RMS_i)
    uy_RMS_dx10.append(uy_RMS_i)
    uz_RMS_dx10.append(uz_RMS_i)
    
    vort_x_dx10.append(vort_x_i)
    vort_y_dx10.append(vort_y_i)
    vort_z_dx10.append(vort_z_i)
    vort_x_mean_dx10.append(vort_x_mean_i)
    vort_y_mean_dx10.append(vort_y_mean_i)
    vort_z_mean_dx10.append(vort_z_mean_i)
    
    TKE_dx10.append(TKE_i)
    vort_mag_dx10.append(vort_mag_i/1e5)
    vort_mean_mag_dx10.append(vort_mean_mag_i/1e5)
    
    
# get data dx20
x_dx20 = []; p_dx20 = []; p_mean_dx20 = []
ux_dx20 = []; uy_dx20 = []; uz_dx20 = []
ux_mean_dx20 = []; uy_mean_dx20 = []; uz_mean_dx20 = []; 
ux_RMS_dx20 = []; uy_RMS_dx20 = []; uz_RMS_dx20 = []; 
vort_x_dx20 = []; vort_y_dx20 = []; vort_z_dx20 = []; 
vort_x_mean_dx20 = []; vort_y_mean_dx20 = []; vort_z_mean_dx20 = [];
TKE_dx20 = []; vort_mag_dx20 = []; vort_mean_mag_dx20 =  []
for i in range(len(dfs_dx20)):
    df = dfs_dx20[i]
    
    x_i    = df['Points_0'].values*1e3/r_inj
    p_i    = df['P'].values*1e3
    p_mean_i = df['P_MEAN'].values*1e3
    ux_i   = df['U_0'].values
    uy_i   = df['U_1'].values
    uz_i   = df['U_2'].values
    ux_mean_i   = df['U_MEAN_0'].values
    uy_mean_i   = df['U_MEAN_1'].values
    uz_mean_i   = df['U_MEAN_2'].values
    ux_RMS_i   = df['U_RMS_0'].values
    uy_RMS_i   = df['U_RMS_1'].values
    uz_RMS_i   = df['U_RMS_2'].values
    vort_x_i = df['VORT_0'].values
    vort_y_i = df['VORT_1'].values
    vort_z_i = df['VORT_2'].values
    vort_x_mean_i = df['VORT_MEAN_0'].values
    vort_y_mean_i = df['VORT_MEAN_1'].values
    vort_z_mean_i = df['VORT_MEAN_2'].values
    
    TKE_i = 0.5*(ux_RMS_i**2+uy_RMS_i**2+uz_RMS_i**2)
    vort_mag_i = np.sqrt(vort_x_i**2+vort_y_i**2+vort_z_i**2)
    vort_mean_mag_i = np.sqrt(vort_x_mean_i**2+vort_y_mean_i**2+vort_z_mean_i**2)
    
    
    x_dx20.append(x_i)
    p_dx20.append(p_i)
    p_mean_dx20.append(p_mean_i)
    
    ux_dx20.append(ux_i)
    uy_dx20.append(uy_i)
    uz_dx20.append(uz_i)
    ux_mean_dx20.append(ux_mean_i)
    uy_mean_dx20.append(uy_mean_i)
    uz_mean_dx20.append(uz_mean_i)
    ux_RMS_dx20.append(ux_RMS_i)
    uy_RMS_dx20.append(uy_RMS_i)
    uz_RMS_dx20.append(uz_RMS_i)
    
    vort_x_dx20.append(vort_x_i)
    vort_y_dx20.append(vort_y_i)
    vort_z_dx20.append(vort_z_i)
    vort_x_mean_dx20.append(vort_x_mean_i)
    vort_y_mean_dx20.append(vort_y_mean_i)
    vort_z_mean_dx20.append(vort_z_mean_i)
    
    TKE_dx20.append(TKE_i)
    vort_mag_dx20.append(vort_mag_i/1e5)
    vort_mean_mag_dx20.append(vort_mean_mag_i/1e5)
    
    
#%% Find BL thickness with uz_mean
perc_u = 0.98

x_BL_dx20 = []; x_BL_dx20_2 = []
x_BL_dx10 = []; x_BL_dx10_2 = []
for i in range(len(labels_z_lines)):

    x_arr = x_dx20[i]
    u_arr = uz_mean_dx20[i]
    u_bulk = u_arr[int(len(u_arr)/2)]
    x_BL_found = False; n = 1
    while (not x_BL_found):
        u_n = u_arr[n]
        if u_arr[n] < perc_u*u_bulk:
            n += 1
        else:
            x_BL_found = True
            x_BL_dx20_zi = x_arr[n]
    
    x_arr = x_dx10[i]
    u_arr = uz_mean_dx10[i]
    u_bulk = u_arr[int(len(u_arr)/2)]
    x_BL_found = False; n = 1
    while (not x_BL_found):
        u_n = u_arr[n]
        if u_arr[n] < perc_u*u_bulk:
            n += 1
        else:
            x_BL_found = True
            x_BL_dx10_zi  = x_arr[n]
    
    x_arr = x_dx20[i]
    u_arr = uz_mean_dx20[i]
    u_bulk = u_arr[int(len(u_arr)/2)]
    x_BL_found = False; n = int(len(u_arr)/2)
    while (not x_BL_found):
        u_n = u_arr[n]
        if u_arr[n] > perc_u*u_bulk:
            n -= 1
        else:
            x_BL_found = True
            x_BL_dx20_2_zi  = x_arr[n]
            
    x_arr = x_dx10[i]
    u_arr = uz_mean_dx10[i]
    u_bulk = u_arr[int(len(u_arr)/2)]
    x_BL_found = False; n = int(len(u_arr)/2)
    while (not x_BL_found):
        u_n = u_arr[n]
        if u_arr[n] > perc_u*u_bulk:
            n -= 1
        else:
            x_BL_found = True
            x_BL_dx10_2_zi  = x_arr[n]
    
    x_BL_dx20.append(x_BL_dx20_zi)
    x_BL_dx20_2.append(x_BL_dx20_2_zi)
    x_BL_dx10.append(x_BL_dx10_zi)
    x_BL_dx10_2.append(x_BL_dx10_2_zi)
        
        
 


#%% Plots over lines


y_lim_vort = (0,4)
y_lim_TKE = (0,0.75)


i = 1

# uz_mean
plt.figure(figsize=figsize_)
plt.title(labels_z_lines[i])
plt.plot(x_dx20[i],uz_mean_dx20[i],'k',label=label_dx20)
plt.plot(x_dx10[i],uz_mean_dx10[i],'b',label=label_dx10)
plt.plot([x_BL_dx20[i]]*2,[0,40],'k--',label=r'$\delta_{\mathrm{DX}20}$')
plt.plot([x_BL_dx10[i]]*2,[0,40],'b--',label=r'$\delta_{\mathrm{DX}10}$')
'''
plt.plot([x_BL_dx10]*2,[0,40],'k--',label='BL1 dx10')
plt.plot([x_BL_dx20]*2,[0,40],'b--',label='BL1 dx20')
plt.plot([x_BL_dx10_2]*2,[0,40],'k',label='BL2 dx10')
plt.plot([x_BL_dx20_2]*2,[0,40],'b',label='BL2 dx20')
'''
#plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_)
plt.ylim(y_lim_uz)
plt.xlabel(label_x)
plt.ylabel(label_u)
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_data_injector_uz_'+labels_save_figs[i]+'.pdf')
plt.show()
plt.close()

# vorticity
plt.figure(figsize=figsize_)
plt.title(labels_z_lines[i])
plt.plot(x_dx20[i],vort_mean_mag_dx20[i],'k',label=label_dx20)
plt.plot(x_dx10[i],vort_mean_mag_dx10[i],'b',label=label_dx10)
plt.plot([x_BL_dx20[i]]*2,[0,5e5],'k--',label=r'$\delta_{\mathrm{DX}20}$')
plt.plot([x_BL_dx10[i]]*2,[0,5e5],'b--',label=r'$\delta_{\mathrm{DX}10}$')
#plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_)
plt.ylim(y_lim_vort)
plt.xlabel(label_x)
plt.ylabel(label_vort)
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_data_injector_vort_'+labels_save_figs[i]+'.pdf')
plt.show()
plt.close()

# TKE
plt.figure(figsize=figsize_)
plt.title(labels_z_lines[i])
plt.plot(x_dx20[i],TKE_dx20[i],'k',label=label_dx20)
plt.plot(x_dx10[i],TKE_dx10[i],'b',label=label_dx10)
plt.plot([x_BL_dx20[i]]*2,[0,5e5],'k--',label=r'$\delta_{\mathrm{DX}20}$')
plt.plot([x_BL_dx10[i]]*2,[0,5e5],'b--',label=r'$\delta_{\mathrm{DX}10}$')
#plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_)
plt.ylim(y_lim_TKE)
plt.xlabel(label_x)
plt.ylabel(label_TKE)
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_data_injector_TKE_'+labels_save_figs[i]+'.pdf')
plt.show()
plt.close()
