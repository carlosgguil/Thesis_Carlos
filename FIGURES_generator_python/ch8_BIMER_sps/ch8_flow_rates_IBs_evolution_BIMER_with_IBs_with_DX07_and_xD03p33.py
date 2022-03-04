# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/flow_rates_ibs/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/IBs/'
sys.path.append(folder)

# Change size of figures if wished
FFIG = 0.5
figsize_ = (FFIG*30,FFIG*20)
figsize_3_in_a_row = (FFIG*45,FFIG*15) #(FFIG*55,FFIG*15)
figsize_bar = (FFIG*50,FFIG*20)

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 70*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['lines.markersize'] = 45*FFIG
plt.rcParams['text.usetex'] = True


# Injected flow rates
SCALE_FACTOR = 1e9
d_inj = 0.45E-3
Q_inj = 185.3 

# Times correspond to x_c/d_inj = 6.67 #10
tau_dr_DX15  = 562e-3 #633e-3
tau_dr_DX10  = 354e-3 #428e-3
tau_dr_DX07p5 = 359e-3 #434e-3



# Define labels and tags
x_label_time   = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
y_label_Ql_inst = r"$Q_l ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_mean_perp = r"$\overline{Q_l} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_RMS_perp = r"$Q_{l,\mathrm{RMS}} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"


label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07  = r'$\mathrm{DX}07$'
cases = [label_DX07 , label_DX10, label_DX15]

label_Ql_injected = r'$Q_l ~\mathrm{injected}$'

label_xD_03p33 = r'$x_c/d_\mathrm{inj} = 3.33$'
label_xD_05p00 = r'$x_c/d_\mathrm{inj} = 5$'
label_xD_06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'
label_xD_08p33 = r'$x_c/d_\mathrm{inj} = 8.33$'
label_xD_10p00 = r'$x_c/d_\mathrm{inj} = 10$'
label_xD_11p66 = r'$x_c/d_\mathrm{inj} = 11.67$'

# For bar graphs
barWidth = 0.25
r1 = np.arange(len(cases))


tp_0_true_values = False

if tp_0_true_values:
    tp_0_DX07 = 0.9310/tau_dr_DX07p5
else:
    tp_0_DX07 = 2.05

# these are ~ 2 anyways
tp_0_DX10 = 0.7688/tau_dr_DX10
tp_0_DX15 = 1.1693/tau_dr_DX15 


# define maximum values for t' (obtained from ch8_nelem_plot.py)
tp_max_DX15 = 6.775423875670118 # diff of 1*tp
tp_max_DX10 = 4.88789371578306 
tp_max_DX07 = 3.9651507666425956 


# define t_min and ticks of mean, RMS evolution graphs
t_min = 2 #min(t_UG100_DX10_x05)
'''
tp_ticks_UG75_DX10 = np.array([0,0.6,1.2])+t_min
tp_ticks_UG75_DX20 = np.array([0,3,6,9,12,15])+t_min
tp_ticks_UG100_DX10 = np.array([0,0.6,1.2])+t_min
tp_ticks_UG100_DX20 = np.array([0,4,8,12,16,20])+t_min
'''

#%% Read iso-x dataframes

df_DX15_x03p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_03p33.csv')
df_DX15_x05p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_05p00.csv')
df_DX15_x06p67 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_06p67.csv')
df_DX15_x08p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_08p33.csv')
df_DX15_x10p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_10p00.csv')
df_DX15_x11p66 = pd.read_csv(folder+'/overall_integrated_fluxes/dx15p0_Q_xD_11p66.csv')


df_DX10_x03p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_03p33.csv')
df_DX10_x05p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_05p00.csv')
df_DX10_x06p67 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_06p67.csv')
df_DX10_x08p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_08p33.csv')
df_DX10_x10p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_10p00.csv')
df_DX10_x11p66 = pd.read_csv(folder+'/overall_integrated_fluxes/dx10p0_Q_xD_11p66.csv')


df_DX07_x03p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_03p33.csv')
df_DX07_x05p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_05p00.csv')
df_DX07_x06p67 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_06p67.csv')
df_DX07_x08p33 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_08p33.csv')
df_DX07_x10p00 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_10p00.csv')
df_DX07_x11p66 = pd.read_csv(folder+'/overall_integrated_fluxes/dx07p5_Q_xD_11p66.csv')

#%% Extract time and Qs from iso-x dataframes

# DX15
t_DX15_x03p33 = df_DX15_x03p33['t_xD_03p33'].values
t_DX15_x03p33 = (t_DX15_x03p33-t_DX15_x03p33[0])/tau_dr_DX15 + tp_0_DX15
Q_inst_DX15_x03p33 = df_DX15_x03p33['Q_t_xD_03p33'].values
Q_mean_DX15_x03p33 = df_DX15_x03p33['Q_t_xD_03p33_mean_evol'].values
Q_RMS_DX15_x03p33  = df_DX15_x03p33['Q_t_xD_03p33_rms_evol'].values

t_DX15_x05p00 = df_DX15_x05p00['t_xD_05p00'].values
t_DX15_x05p00 = (t_DX15_x05p00-t_DX15_x05p00[0])/tau_dr_DX15 + tp_0_DX15
Q_inst_DX15_x05p00 = df_DX15_x05p00['Q_t_xD_05p00'].values
Q_mean_DX15_x05p00 = df_DX15_x05p00['Q_t_xD_05p00_mean_evol'].values
Q_RMS_DX15_x05p00  = df_DX15_x05p00['Q_t_xD_05p00_rms_evol'].values


t_DX15_x06p67 = df_DX15_x06p67['t_xD_06p67'].values
t_DX15_x06p67 = (t_DX15_x06p67-t_DX15_x06p67[0])/tau_dr_DX15 + tp_0_DX15
Q_mean_DX15_x06p67 = df_DX15_x06p67['Q_t_xD_06p67_mean_evol'].values
Q_RMS_DX15_x06p67  = df_DX15_x06p67['Q_t_xD_06p67_rms_evol'].values

t_DX15_x08p33 = df_DX15_x08p33['t_xD_08p33'].values
t_DX15_x08p33 = (t_DX15_x08p33-t_DX15_x08p33[0])/tau_dr_DX15 + tp_0_DX15
Q_mean_DX15_x08p33 = df_DX15_x08p33['Q_t_xD_08p33_mean_evol'].values
Q_RMS_DX15_x08p33  = df_DX15_x08p33['Q_t_xD_08p33_rms_evol'].values

t_DX15_x10p00 = df_DX15_x10p00['t_xD_10p00'].values
t_DX15_x10p00 = (t_DX15_x10p00-t_DX15_x10p00[0])/tau_dr_DX15 + tp_0_DX15
Q_mean_DX15_x10p00 = df_DX15_x10p00['Q_t_xD_10p00_mean_evol'].values
Q_RMS_DX15_x10p00  = df_DX15_x10p00['Q_t_xD_10p00_rms_evol'].values

t_DX15_x11p66 = df_DX15_x11p66['t_xD_11p66'].values
t_DX15_x11p66 = (t_DX15_x11p66-t_DX15_x11p66[0])/tau_dr_DX15 + tp_0_DX15
Q_inst_DX15_x11p66 = df_DX15_x11p66['Q_t_xD_11p66'].values
Q_mean_DX15_x11p66 = df_DX15_x11p66['Q_t_xD_11p66_mean_evol'].values
Q_RMS_DX15_x11p66  = df_DX15_x11p66['Q_t_xD_11p66_rms_evol'].values



# DX10
t_DX10_x03p33 = df_DX10_x03p33['t_xD_03p33'].values
t_DX10_x03p33 = (t_DX10_x03p33-t_DX10_x03p33[0])/tau_dr_DX10 + tp_0_DX10
Q_inst_DX10_x03p33 = df_DX10_x03p33['Q_t_xD_03p33'].values
Q_mean_DX10_x03p33 = df_DX10_x03p33['Q_t_xD_03p33_mean_evol'].values
Q_RMS_DX10_x03p33  = df_DX10_x03p33['Q_t_xD_03p33_rms_evol'].values

t_DX10_x05p00 = df_DX10_x05p00['t_xD_05p00'].values
t_DX10_x05p00 = (t_DX10_x05p00-t_DX10_x05p00[0])/tau_dr_DX10 + tp_0_DX10
Q_inst_DX10_x05p00 = df_DX10_x05p00['Q_t_xD_05p00'].values
Q_mean_DX10_x05p00 = df_DX10_x05p00['Q_t_xD_05p00_mean_evol'].values
Q_RMS_DX10_x05p00  = df_DX10_x05p00['Q_t_xD_05p00_rms_evol'].values


t_DX10_x06p67 = df_DX10_x06p67['t_xD_06p67'].values
t_DX10_x06p67 = (t_DX10_x06p67-t_DX10_x06p67[0])/tau_dr_DX10 + tp_0_DX10
Q_inst_DX10_x06p67 = df_DX10_x06p67['Q_t_xD_06p67'].values
Q_mean_DX10_x06p67 = df_DX10_x06p67['Q_t_xD_06p67_mean_evol'].values
Q_RMS_DX10_x06p67  = df_DX10_x06p67['Q_t_xD_06p67_rms_evol'].values

t_DX10_x08p33 = df_DX10_x08p33['t_xD_08p33'].values
t_DX10_x08p33 = (t_DX10_x08p33-t_DX10_x08p33[0])/tau_dr_DX10 + tp_0_DX10
Q_mean_DX10_x08p33 = df_DX10_x08p33['Q_t_xD_08p33_mean_evol'].values
Q_RMS_DX10_x08p33  = df_DX10_x08p33['Q_t_xD_08p33_rms_evol'].values

t_DX10_x10p00 = df_DX10_x10p00['t_xD_10p00'].values
t_DX10_x10p00 = (t_DX10_x10p00-t_DX10_x10p00[0])/tau_dr_DX10 + tp_0_DX10
Q_mean_DX10_x10p00 = df_DX10_x10p00['Q_t_xD_10p00_mean_evol'].values
Q_RMS_DX10_x10p00  = df_DX10_x10p00['Q_t_xD_10p00_rms_evol'].values

t_DX10_x11p66 = df_DX10_x11p66['t_xD_11p66'].values
t_DX10_x11p66 = (t_DX10_x11p66-t_DX10_x11p66[0])/tau_dr_DX10 + tp_0_DX10
Q_inst_DX10_x11p66 = df_DX10_x11p66['Q_t_xD_11p66'].values
Q_mean_DX10_x11p66 = df_DX10_x11p66['Q_t_xD_11p66_mean_evol'].values
Q_RMS_DX10_x11p66  = df_DX10_x11p66['Q_t_xD_11p66_rms_evol'].values


# DX07
t_DX07_x03p33 = df_DX07_x03p33['t_xD_03p33'].values
t_DX07_x03p33 = (t_DX07_x03p33-t_DX07_x03p33[0])/tau_dr_DX07p5 + tp_0_DX07
Q_inst_DX07_x03p33 = df_DX07_x03p33['Q_t_xD_03p33'].values
Q_mean_DX07_x03p33 = df_DX07_x03p33['Q_t_xD_03p33_mean_evol'].values
Q_RMS_DX07_x03p33  = df_DX07_x03p33['Q_t_xD_03p33_rms_evol'].values

t_DX07_x05p00 = df_DX07_x05p00['t_xD_05p00'].values
t_DX07_x05p00 = (t_DX07_x05p00-t_DX07_x05p00[0])/tau_dr_DX07p5 + tp_0_DX07
Q_inst_DX07_x05p00 = df_DX07_x05p00['Q_t_xD_05p00'].values
Q_mean_DX07_x05p00 = df_DX07_x05p00['Q_t_xD_05p00_mean_evol'].values
Q_RMS_DX07_x05p00  = df_DX07_x05p00['Q_t_xD_05p00_rms_evol'].values


t_DX07_x06p67 = df_DX07_x06p67['t_xD_06p67'].values
t_DX07_x06p67 = (t_DX07_x06p67-t_DX07_x06p67[0])/tau_dr_DX07p5 + tp_0_DX07
Q_mean_DX07_x06p67 = df_DX07_x06p67['Q_t_xD_06p67_mean_evol'].values
Q_RMS_DX07_x06p67  = df_DX07_x06p67['Q_t_xD_06p67_rms_evol'].values

t_DX07_x08p33 = df_DX07_x08p33['t_xD_08p33'].values
t_DX07_x08p33 = (t_DX07_x08p33-t_DX07_x08p33[0])/tau_dr_DX07p5 + tp_0_DX07
Q_mean_DX07_x08p33 = df_DX07_x08p33['Q_t_xD_08p33_mean_evol'].values
Q_RMS_DX07_x08p33  = df_DX07_x08p33['Q_t_xD_08p33_rms_evol'].values

t_DX07_x10p00 = df_DX07_x10p00['t_xD_10p00'].values
t_DX07_x10p00 = (t_DX07_x10p00-t_DX07_x10p00[0])/tau_dr_DX07p5 + tp_0_DX07
Q_mean_DX07_x10p00 = df_DX07_x10p00['Q_t_xD_10p00_mean_evol'].values
Q_RMS_DX07_x10p00  = df_DX07_x10p00['Q_t_xD_10p00_rms_evol'].values

t_DX07_x11p66 = df_DX07_x11p66['t_xD_11p66'].values
t_DX07_x11p66 = (t_DX07_x11p66-t_DX07_x11p66[0])/tau_dr_DX07p5 + tp_0_DX07
Q_inst_DX07_x11p66 = df_DX07_x11p66['Q_t_xD_11p66'].values
Q_mean_DX07_x11p66 = df_DX07_x11p66['Q_t_xD_11p66_mean_evol'].values
Q_RMS_DX07_x11p66  = df_DX07_x11p66['Q_t_xD_11p66_rms_evol'].values



#%% Transform times  (only DX07)


# DX07
m_DX07 = (tp_max_DX07 - tp_0_DX07)/(t_DX07_x03p33[-1] - t_DX07_x03p33[0])
t_DX07 = m_DX07*(t_DX07_x03p33 - t_DX07_x03p33[0]) + tp_0_DX07
#t_UG100_DX20_x10 = m_op1_dx20*(t_UG100_DX20_x10 - t_UG100_DX20_x10[0]) + tp_0_UG100_DX20
#t_UG100_DX20_x10 = m_op1_dx20*(t_UG100_DX20_x15 - t_UG100_DX20_x15[0]) + tp_0_UG100_DX20
t_DX07_x03p33 = t_DX07
t_DX07_x05p00 = t_DX07
t_DX07_x06p67 = t_DX07
t_DX07_x08p33 = t_DX07
t_DX07_x10p00 = t_DX07
t_DX07_x11p66 = t_DX07



#%% Plot time evolution of instantaneous Qs 

#t_max = max(t_UG100_DX10_x05)

# DX15
plt.figure(figsize=figsize_)
plt.title(label_DX15)
plt.plot(t_DX15_x03p33, Q_inst_DX15_x03p33, 'b', label=label_xD_03p33)
plt.plot(t_DX15_x05p00, Q_inst_DX15_x05p00, 'k', label=label_xD_05p00)
plt.plot(t_DX15_x11p66, Q_inst_DX15_x11p66, 'r', label=label_xD_11p66)
#plt.plot([t_min, t_max], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
#plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_inst)
#plt.xlim(t_min-0.05,t_max+0.05)
#plt.ylim(0,1e4)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10.pdf')
plt.show()
plt.close()

#%% Instantaneous rate at DX10

t_min = min(t_DX10_x03p33)
t_max = max(t_DX10_x03p33)
# DX10
plt.figure(figsize=figsize_)
plt.title(label_DX10)
plt.plot(t_DX10_x03p33, Q_inst_DX10_x03p33, 'b', label=label_xD_03p33)
plt.plot(t_DX10_x05p00, Q_inst_DX10_x05p00, 'k', label=label_xD_05p00)
plt.plot(t_DX10_x06p67, Q_inst_DX10_x06p67, 'r', label=label_xD_06p67)
#plt.plot(t_DX10_x11p66, Q_inst_DX10_x11p66, 'r', label=label_xD_11p66)
plt.plot([t_min, t_max], [Q_inj]*2, '--k', label=label_Ql_injected)
plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_inst)
plt.xlim(t_min-0.05,t_max+0.05)
plt.ylim(0,600)
plt.legend(loc='best',fontsize=60*FFIG,ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'inst_Q_iso_x_DX10.pdf')
plt.show()
plt.close()

#%%-
# DX07p5
plt.figure(figsize=figsize_)
plt.title(label_DX07)
plt.plot(t_DX07_x03p33, Q_inst_DX07_x03p33, 'b', label=label_xD_03p33)
plt.plot(t_DX07_x05p00, Q_inst_DX07_x05p00, 'k', label=label_xD_05p00)
plt.plot(t_DX07_x11p66, Q_inst_DX07_x11p66, 'g', label=label_xD_11p66)
#plt.plot([t_min, t_max], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
#plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_inst)
#plt.xlim(t_min-0.05,t_max+0.05)
#plt.ylim(0,1e4)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10.pdf')
plt.show()
plt.close()





label_xD_03p33 = r'$x_c/d_\mathrm{inj} = 3.33$'
label_xD_05p00 = r'$x_c/d_\mathrm{inj} = 5$'
label_xD_06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'
label_xD_08p33 = r'$x_c/d_\mathrm{inj} = 8.33$'
label_xD_10p00 = r'$x_c/d_\mathrm{inj} = 10$'
label_xD_11p67 = r'$x_c/d_\mathrm{inj} = 11.67$'

#%% Mean values evolution in iso-x planes



fig = plt.figure(figsize=figsize_3_in_a_row)
gs = fig.add_gridspec(1, 3, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3) = gs.subplots(sharey='row')
ax1.plot([t_min, max(t_DX07_x03p33)], [Q_inj]*2, '--k', label=label_Ql_injected)
ax1.plot(t_DX07_x03p33, Q_mean_DX07_x03p33, 'b', label=label_xD_03p33)
ax1.plot(t_DX07_x05p00, Q_mean_DX07_x05p00, 'k', label=label_xD_05p00)
ax1.plot(t_DX07_x06p67, Q_mean_DX07_x06p67, 'r', label=label_xD_06p67)
#ax1.plot(t_DX07_x08p33, Q_mean_DX07_x08p33, '--b', label=label_xD_08p33)
#ax1.plot(t_DX07_x10p00, Q_mean_DX07_x10p00, '--k', label=label_xD_10p00)
#ax1.plot(t_DX07_x11p66, Q_mean_DX07_x11p66, '--r', label=label_xD_11p66)
ax1.set_title(label_DX07)
#ax1.xaxis.set_ticks(tp_ticks_UG75_DX10)
#ax1.yaxis.set_ticks([0,1000,2000,3000,4000,5000])

ax2.plot([t_min, max(t_DX10_x03p33)], [Q_inj]*2, '--k', label=label_Ql_injected)
ax2.plot(t_DX10_x03p33, Q_mean_DX10_x03p33, 'b', label=label_xD_03p33)
ax2.plot(t_DX10_x05p00, Q_mean_DX10_x05p00, 'k', label=label_xD_05p00)
ax2.plot(t_DX10_x06p67, Q_mean_DX10_x06p67, 'r', label=label_xD_06p67)
#ax2.plot(t_DX10_x08p33, Q_mean_DX10_x08p33, '--b', label=label_xD_08p33)
#ax2.plot(t_DX10_x10p00, Q_mean_DX10_x10p00, '--k', label=label_xD_10p00)
#ax2.plot(t_DX10_x11p66, Q_mean_DX10_x11p66, '--r', label=label_xD_11p66)
ax2.set_title(label_DX10)
#ax2.set_title(label_UG75_DX20)
#ax2.xaxis.set_ticks(tp_ticks_UG75_DX20)
ax2.legend(loc='best',fontsize=45*FFIG,ncol=2)



ax3.plot([t_min, max(t_DX15_x03p33)], [Q_inj]*2, '--k', label=label_Ql_injected)
ax3.plot(t_DX15_x03p33, Q_mean_DX15_x03p33, 'b', label=label_xD_03p33)
ax3.plot(t_DX15_x05p00, Q_mean_DX15_x05p00, 'k', label=label_xD_05p00)
ax3.plot(t_DX15_x06p67, Q_mean_DX15_x06p67, 'r', label=label_xD_06p67)
#ax3.plot(t_DX15_x08p33, Q_mean_DX15_x08p33, '--b', label=label_xD_08p33)
#ax3.plot(t_DX15_x10p00, Q_mean_DX15_x10p00, '--k', label=label_xD_10p00)
#ax3.plot(t_DX15_x11p66, Q_mean_DX15_x11p66, '--r', label=label_xD_11p66)
ax3.set_title(label_DX15)
#ax3.xaxis.set_ticks(tp_ticks_UG100_DX10)

axs.flat[0].set(ylabel = y_label_Ql_mean_perp)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,300)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_mean_Q_iso_x.pdf')
plt.show()
plt.close


#%% RMS values evolution in iso-x planes



fig = plt.figure(figsize=figsize_3_in_a_row)
gs = fig.add_gridspec(1, 3, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3) = gs.subplots(sharey='row')
ax1.plot(t_DX07_x03p33, Q_RMS_DX07_x03p33, 'b', label=label_xD_03p33)
ax1.plot(t_DX07_x05p00, Q_RMS_DX07_x05p00, 'k', label=label_xD_05p00)
ax1.plot(t_DX07_x06p67, Q_RMS_DX07_x06p67, 'r', label=label_xD_06p67)
#ax1.plot(t_DX07_x08p33, Q_RMS_DX07_x08p33, '--b', label=label_xD_08p33)
#ax1.plot(t_DX07_x10p00, Q_RMS_DX07_x10p00, '--k', label=label_xD_10p00)
#ax1.plot(t_DX07_x11p66, Q_RMS_DX07_x11p66, '--r', label=label_xD_11p66)
ax1.set_title(label_DX07)
#ax1.xaxis.set_ticks(tp_ticks_UG75_DX10)
#ax1.yaxis.set_ticks([0,1000,2000,3000,4000,5000])

ax2.plot(t_DX10_x03p33, Q_RMS_DX10_x03p33, 'b', label=label_xD_03p33)
ax2.plot(t_DX10_x05p00, Q_RMS_DX10_x05p00, 'k', label=label_xD_05p00)
ax2.plot(t_DX10_x06p67, Q_RMS_DX10_x06p67, 'r', label=label_xD_06p67)
#ax2.plot(t_DX10_x08p33, Q_RMS_DX10_x08p33, '--b', label=label_xD_08p33)
#ax2.plot(t_DX10_x10p00, Q_RMS_DX10_x10p00, '--k', label=label_xD_10p00)
#ax2.plot(t_DX10_x11p66, Q_RMS_DX10_x11p66, '--r', label=label_xD_11p66)
ax2.set_title(label_DX10)
#ax2.set_title(label_UG75_DX20)
#ax2.xaxis.set_ticks(tp_ticks_UG75_DX20)
#ax2.legend(loc='best',fontsize=45*FFIG,ncol=2)



ax3.plot(t_DX15_x03p33, Q_RMS_DX15_x03p33, 'b', label=label_xD_03p33)
ax3.plot(t_DX15_x05p00, Q_RMS_DX15_x05p00, 'k', label=label_xD_05p00)
ax3.plot(t_DX15_x06p67, Q_RMS_DX15_x06p67, 'r', label=label_xD_06p67)
#ax3.plot(t_DX15_x08p33, Q_RMS_DX15_x08p33, '--b', label=label_xD_08p33)
#ax3.plot(t_DX15_x10p00, Q_RMS_DX15_x10p00, '--k', label=label_xD_10p00)
#ax3.plot(t_DX15_x11p66, Q_RMS_DX15_x11p66, '--r', label=label_xD_11p66)
ax3.set_title(label_DX15)
#ax3.xaxis.set_ticks(tp_ticks_UG100_DX10)

axs.flat[0].set(ylabel = y_label_Ql_RMS_perp)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,250)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_RMS_Q_iso_x.pdf')
plt.show()
plt.close



#%% Bar graphs iso-x

Q_x_mean_xD_03p33 = [Q_mean_DX07_x03p33[-1],Q_mean_DX10_x03p33[-1],
                     Q_mean_DX15_x03p33[-1]]
Q_x_mean_xD_05p00 = [Q_mean_DX07_x05p00[-1],Q_mean_DX10_x05p00[-1],
                     Q_mean_DX15_x05p00[-1]]
Q_x_mean_xD_06p67 = [Q_mean_DX07_x06p67[-1],Q_mean_DX10_x06p67[-1],
                     Q_mean_DX15_x06p67[-1]]

Q_x_RMS_xD_03p33 = [Q_RMS_DX07_x03p33[-1],Q_RMS_DX10_x03p33[-1],
                     Q_RMS_DX15_x03p33[-1]]
Q_x_RMS_xD_05p00 = [Q_RMS_DX07_x05p00[-1],Q_RMS_DX10_x05p00[-1],
                     Q_RMS_DX15_x05p00[-1]]
Q_x_RMS_xD_06p67 = [Q_RMS_DX07_x06p67[-1],Q_RMS_DX10_x06p67[-1],
                     Q_RMS_DX15_x06p67[-1]]





# Bar graph with RMS
plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.plot([r1[0]-barWidth*1.5,r1[-1]+barWidth*1.5],[Q_inj]*2, '--k', label=label_Ql_injected,linewidth=4*FFIG)
plt.bar(r1-0.25, Q_x_mean_xD_03p33, yerr=Q_x_RMS_xD_03p33, width=barWidth, color='blue', edgecolor='white', label=label_xD_03p33, capsize=barWidth*20)
plt.bar(r1, Q_x_mean_xD_05p00, yerr=Q_x_RMS_xD_05p00, width=barWidth, color='grey', edgecolor='white', label=label_xD_05p00, capsize=barWidth*20)
plt.bar(r1+0.25, Q_x_mean_xD_06p67, yerr=Q_x_RMS_xD_06p67, width=barWidth, color='red', edgecolor='white', label=label_xD_06p67, capsize=barWidth*20)
#plt.xlabel('Case')#, fontweight='bold')
plt.ylabel(y_label_Ql_mean_perp)
plt.xticks([r for r in range(len(cases))], cases)
plt.legend(loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'bar_graph_isox_IBs.pdf')
plt.show()
plt.close()


