# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/flow_rates_ibs/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/IBs/'
sys.path.append(folder)

# Change size of figures if wished
FFIG = 0.5
figsize_ = (FFIG*30,FFIG*20)
figsize_4_in_a_row = (FFIG*55,FFIG*15)
figsize_bar = (FFIG*50,FFIG*20)

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['lines.markersize'] = 45*FFIG
plt.rcParams['text.usetex'] = True


# Injected flow rates
SCALE_FACTOR = 1e9
d_inj = 0.45E-3
Q_inj_UG100 = np.pi/4*d_inj**2*23.33*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR
Q_inj_UG75 = np.pi/4*d_inj**2*17.5*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR

# Define characteristic times
tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584

# Define labels and tags
x_label_time   = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
y_label_Ql_inst = r"$Q_l ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_mean_perp = r"$\overline{Q_l}_{,\mathrm{perp}} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_mean_film = r"$\overline{Q_l}_{,\mathrm{film}} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_RMS_perp = r"$Q_{l,\mathrm{RMS},\mathrm{perp}} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"
y_label_Ql_RMS_film = r"$Q_{l,\mathrm{RMS},\mathrm{film}} ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"


label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
cases = [label_UG75_DX10 , label_UG75_DX20,
         label_UG100_DX10, label_UG100_DX20]

label_Ql_injected = r'$Q_l ~\mathrm{injected}$'

label_x_equal_5  = r'$x = 5~\mathrm{mm}$'
label_x_equal_10 = r'$x = 10~\mathrm{mm}$'
label_x_equal_15 = r'$x = 15~\mathrm{mm}$'

label_x_less_5  = r'$x < 5~\mathrm{mm}$'
label_x_less_10 = r'$x < 10~\mathrm{mm}$'
label_x_less_15 = r'$x < 15~\mathrm{mm}$'

# For bar graphs
barWidth = 0.25
r1 = np.arange(len(cases))
r2 = np.array([1,3])


#%% Read iso-x dataframes

df_UG100_DX20_x05 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_x05')
df_UG100_DX20_x10 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_x10')
df_UG100_DX20_x15 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_x15')

df_UG100_DX10_x05 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_x05')
df_UG100_DX10_x10 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_x10')

df_UG75_DX20_x05 = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_x05')
df_UG75_DX20_x10 = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_x10')
df_UG75_DX20_x15 = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_x15')

df_UG75_DX10_x05 = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx10_Q_x05')
df_UG75_DX10_x10 = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx10_Q_x10')

# Read filming dataframes
df_UG100_DX20_x05_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_film_x05')
df_UG100_DX20_x10_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_film_x10')
df_UG100_DX20_x15_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx20_Q_film_x15')

df_UG100_DX10_x05_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_film_x05')
df_UG100_DX10_x10_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_film_x10')

df_UG75_DX20_x05_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_film_x05')
df_UG75_DX20_x10_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_film_x10')
df_UG75_DX20_x15_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx20_Q_film_x15')

df_UG75_DX10_x05_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx10_Q_film_x05')
df_UG75_DX10_x10_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG75_dx10_Q_film_x10')

#%% Extract time and Qs from iso-x dataframes

# UG100_DX20
t_UG100_DX20_x05 = df_UG100_DX20_x05['t_x05'].values
t_UG100_DX20_x05 = (t_UG100_DX20_x05-t_UG100_DX20_x05[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x05 = df_UG100_DX20_x05['Q_t_x05_mean_evol'].values
Q_RMS_UG100_DX20_x05 = df_UG100_DX20_x05['Q_t_x05_rms_evol'].values


t_UG100_DX20_x10 = df_UG100_DX20_x10['t_x10'].values
t_UG100_DX20_x10 = (t_UG100_DX20_x10-t_UG100_DX20_x10[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x10 = df_UG100_DX20_x10['Q_t_x10_mean_evol'].values
Q_RMS_UG100_DX20_x10 = df_UG100_DX20_x10['Q_t_x10_rms_evol'].values


t_UG100_DX20_x15 = df_UG100_DX20_x15['t_x15'].values
t_UG100_DX20_x15 = (t_UG100_DX20_x15-t_UG100_DX20_x15[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x15 = df_UG100_DX20_x15['Q_t_x15_mean_evol'].values
Q_RMS_UG100_DX20_x15 = df_UG100_DX20_x15['Q_t_x15_rms_evol'].values


# UG100_DX10
t_UG100_DX10_x05 = df_UG100_DX10_x05['t_x05'].values
t_UG100_DX10_x05 = (t_UG100_DX10_x05-t_UG100_DX10_x05[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x05 = df_UG100_DX10_x05['Q_t_x05'].values
Q_mean_UG100_DX10_x05 = df_UG100_DX10_x05['Q_t_x05_mean_evol'].values
Q_RMS_UG100_DX10_x05 = df_UG100_DX10_x05['Q_t_x05_rms_evol'].values


t_UG100_DX10_x10 = df_UG100_DX10_x10['t_x10'].values
t_UG100_DX10_x10 = (t_UG100_DX10_x10-t_UG100_DX10_x10[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x10 = df_UG100_DX10_x10['Q_t_x10'].values
Q_mean_UG100_DX10_x10 = df_UG100_DX10_x10['Q_t_x10_mean_evol'].values
Q_RMS_UG100_DX10_x10 = df_UG100_DX10_x10['Q_t_x10_rms_evol'].values

# UG75_DX20
t_UG75_DX20_x05 = df_UG75_DX20_x05['t_x05'].values
t_UG75_DX20_x05 = (t_UG75_DX20_x05-t_UG75_DX20_x05[0])/tau_ph_UG75_DX20
Q_mean_UG75_DX20_x05 = df_UG75_DX20_x05['Q_t_x05_mean_evol'].values
Q_RMS_UG75_DX20_x05  = df_UG75_DX20_x05['Q_t_x05_rms_evol'].values

t_UG75_DX20_x10 = df_UG75_DX20_x10['t_x10'].values
t_UG75_DX20_x10 = (t_UG75_DX20_x10-t_UG75_DX20_x10[0])/tau_ph_UG75_DX20
Q_mean_UG75_DX20_x10 = df_UG75_DX20_x10['Q_t_x10_mean_evol'].values
Q_RMS_UG75_DX20_x10  = df_UG75_DX20_x10['Q_t_x10_rms_evol'].values

t_UG75_DX20_x15 = df_UG75_DX20_x15['t_x15'].values
t_UG75_DX20_x15 = (t_UG75_DX20_x15-t_UG75_DX20_x15[0])/tau_ph_UG75_DX20
Q_mean_UG75_DX20_x15 = df_UG75_DX20_x15['Q_t_x15_mean_evol'].values
Q_RMS_UG75_DX20_x15  = df_UG75_DX20_x15['Q_t_x15_rms_evol'].values

# UG75_DX10
t_UG75_DX10_x05 = df_UG75_DX10_x05['t_x05'].values
t_UG75_DX10_x05 = (t_UG75_DX10_x05-t_UG75_DX10_x05[0])/tau_ph_UG75_DX10
Q_mean_UG75_DX10_x05 = df_UG75_DX10_x05['Q_t_x05_mean_evol'].values
Q_RMS_UG75_DX10_x05  = df_UG75_DX10_x05['Q_t_x05_rms_evol'].values

t_UG75_DX10_x10 = df_UG75_DX10_x10['t_x10'].values
t_UG75_DX10_x10 = (t_UG75_DX10_x10-t_UG75_DX10_x10[0])/tau_ph_UG75_DX10
Q_mean_UG75_DX10_x10 = df_UG75_DX10_x10['Q_t_x10_mean_evol'].values
Q_RMS_UG75_DX10_x10  = df_UG75_DX10_x10['Q_t_x10_rms_evol'].values



#%% Extract time and Qs from filming dataframes

# UG100_DX20
t_UG100_DX20_x05_filming = df_UG100_DX20_x05_filming['t_film_x05'].values
t_UG100_DX20_x05_filming = (t_UG100_DX20_x05_filming-t_UG100_DX20_x05_filming[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x05_filming = df_UG100_DX20_x05_filming['Q_t_film_x05_mean_evol'].values
Q_RMS_UG100_DX20_x05_filming  = df_UG100_DX20_x05_filming['Q_t_film_x05_rms_evol'].values

t_UG100_DX20_x10_filming = df_UG100_DX20_x10_filming['t_film_x10'].values
t_UG100_DX20_x10_filming = (t_UG100_DX20_x10_filming-t_UG100_DX20_x10_filming[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x10_filming = df_UG100_DX20_x10_filming['Q_t_film_x10_mean_evol'].values
Q_RMS_UG100_DX20_x10_filming  = df_UG100_DX20_x10_filming['Q_t_film_x10_rms_evol'].values

t_UG100_DX20_x15_filming = df_UG100_DX20_x15_filming['t_film_x15'].values
t_UG100_DX20_x15_filming = (t_UG100_DX20_x15_filming-t_UG100_DX20_x15_filming[0])/tau_ph_UG100_DX20
Q_mean_UG100_DX20_x15_filming = df_UG100_DX20_x15_filming['Q_t_film_x15_mean_evol'].values
Q_RMS_UG100_DX20_x15_filming  = df_UG100_DX20_x15_filming['Q_t_film_x15_rms_evol'].values

# UG100_DX10
t_UG100_DX10_x05_filming = df_UG100_DX10_x05_filming['t_film_x05'].values
t_UG100_DX10_x05_filming = (t_UG100_DX10_x05_filming-t_UG100_DX10_x05_filming[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x05_filming = df_UG100_DX10_x05_filming['Q_t_film_x05'].values*SCALE_FACTOR
Q_mean_UG100_DX10_x05_filming = df_UG100_DX10_x05_filming['Q_t_film_x05_mean_evol'].values
Q_RMS_UG100_DX10_x05_filming  = df_UG100_DX10_x05_filming['Q_t_film_x05_rms_evol'].values

t_UG100_DX10_x10_filming = df_UG100_DX10_x10_filming['t_film_x10'].values
t_UG100_DX10_x10_filming = (t_UG100_DX10_x10_filming-t_UG100_DX10_x10_filming[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x10_filming = df_UG100_DX10_x10_filming['Q_t_film_x10'].values*SCALE_FACTOR
Q_mean_UG100_DX10_x10_filming = df_UG100_DX10_x10_filming['Q_t_film_x10_mean_evol'].values
Q_RMS_UG100_DX10_x10_filming  = df_UG100_DX10_x10_filming['Q_t_film_x10_rms_evol'].values

# UG75_DX20
t_UG75_DX20_x05_filming = df_UG75_DX20_x05_filming['t_film_x05'].values
t_UG75_DX20_x05_filming = (t_UG75_DX20_x05_filming-t_UG75_DX20_x05_filming[0])/tau_ph_UG75_DX20
Q_mean_UG75_DX20_x05_filming = df_UG75_DX20_x05_filming['Q_t_film_x05_mean_evol'].values
Q_RMS_UG75_DX20_x05_filming  = df_UG75_DX20_x05_filming['Q_t_film_x05_rms_evol'].values

t_UG75_DX20_x10_filming = df_UG75_DX20_x10_filming['t_film_x10'].values
t_UG75_DX20_x10_filming = (t_UG75_DX20_x10_filming-t_UG75_DX20_x10_filming[0])/tau_ph_UG75_DX20
Q_inst_UG75_DX20_x10_filming = df_UG75_DX20_x10_filming['Q_t_film_x10'].values*SCALE_FACTOR
Q_mean_UG75_DX20_x10_filming = df_UG75_DX20_x10_filming['Q_t_film_x10_mean_evol'].values
Q_RMS_UG75_DX20_x10_filming  = df_UG75_DX20_x10_filming['Q_t_film_x10_rms_evol'].values


t_UG75_DX20_x15_filming = df_UG75_DX20_x15_filming['t_film_x15'].values
t_UG75_DX20_x15_filming = (t_UG75_DX20_x15_filming-t_UG75_DX20_x15_filming[0])/tau_ph_UG75_DX20
Q_mean_UG75_DX20_x15_filming = df_UG75_DX20_x15_filming['Q_t_film_x15_mean_evol'].values
Q_RMS_UG75_DX20_x15_filming  = df_UG75_DX20_x15_filming['Q_t_film_x15_rms_evol'].values


# UG75_DX10
t_UG75_DX10_x05_filming = df_UG75_DX10_x05_filming['t_film_x05'].values
t_UG75_DX10_x05_filming = (t_UG75_DX10_x05_filming-t_UG75_DX10_x05_filming[0])/tau_ph_UG75_DX10
Q_mean_UG75_DX10_x05_filming = df_UG75_DX10_x05_filming['Q_t_film_x05_mean_evol'].values
Q_RMS_UG75_DX10_x05_filming  = df_UG75_DX10_x05_filming['Q_t_film_x05_rms_evol'].values

t_UG75_DX10_x10_filming = df_UG75_DX10_x10_filming['t_film_x10'].values
t_UG75_DX10_x10_filming = (t_UG75_DX10_x10_filming-t_UG75_DX10_x10_filming[0])/tau_ph_UG75_DX10
Q_mean_UG75_DX10_x10_filming = df_UG75_DX10_x10_filming['Q_t_film_x10_mean_evol'].values
Q_RMS_UG75_DX10_x10_filming  = df_UG75_DX10_x10_filming['Q_t_film_x10_rms_evol'].values

#%% Plot time evolution of instantaneous Qs for UG100_DX10

t_min = 0
t_max = max(t_UG100_DX10_x05)

# Iso-x Qs
plt.figure(figsize=figsize_)
#plt.title(r"$Q_l$ evolution ")
plt.plot(t_UG100_DX10_x05, Q_inst_UG100_DX10_x05, 'b', label=label_x_equal_5)
plt.plot(t_UG100_DX10_x10, Q_inst_UG100_DX10_x10, 'k', label=label_x_equal_10)
plt.plot([t_min, t_max], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_inst)
plt.xlim(0-0.05,t_max+0.05)
plt.ylim(0,1e4)
plt.legend(loc='upper center')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10.pdf')
plt.show()
plt.close

# Filming Qs
plt.figure(figsize=figsize_)
#plt.title(r"$Q_l$ evolution ")
plt.plot(t_UG100_DX10_x05_filming, Q_inst_UG100_DX10_x05_filming, 'b', label=label_x_less_5)
plt.plot(t_UG100_DX10_x10_filming, Q_inst_UG100_DX10_x10_filming, 'k', label=label_x_less_10)
plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_inst)
plt.xlim(0-0.05,t_max+0.05)
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10_filming.pdf')
plt.show()
plt.close

#%% [TEST] Plot mean evolution for UG100_DX20

# Iso-x mean Qs
plt.figure(figsize=figsize_)
#plt.title(r"$Q_l$ evolution ")
plt.plot(t_UG100_DX20_x05, Q_mean_UG100_DX20_x05, 'b', label=label_x_equal_5)
plt.plot(t_UG100_DX20_x10, Q_mean_UG100_DX20_x10, 'k', label=label_x_equal_10)
plt.plot(t_UG100_DX20_x10, Q_mean_UG100_DX20_x15, 'r', label=label_x_equal_15)
plt.plot([t_min, max(t_UG100_DX20_x10)], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql_mean_perp)
#plt.ylim(0,1e4)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10.pdf')
plt.show()
plt.close



#%% Mean values evolution in iso-x planes

fig = plt.figure(figsize=figsize_4_in_a_row)
gs = fig.add_gridspec(1, 4, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3, ax4) = gs.subplots(sharey='row')
ax1.plot([t_min, max(t_UG75_DX10_x05)], [Q_inj_UG75]*2, '--k', label=label_Ql_injected)
ax1.plot(t_UG75_DX10_x05, Q_mean_UG75_DX10_x05, 'b', label=label_x_equal_5)
ax1.plot(t_UG75_DX10_x10, Q_mean_UG75_DX10_x10, 'k', label=label_x_equal_10)
#ax1.text(0.5,6000,r'$\mathrm{UG}100\_\mathrm{DX}20$',fontsize=80*FFIG)
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(label_UG75_DX10)
ax1.xaxis.set_ticks([0,1,2,3])
ax1.yaxis.set_ticks([0,1000,2000,3000,4000,5000])

ax2.plot([t_min, max(t_UG75_DX20_x05)], [Q_inj_UG75]*2, '--k', label=label_Ql_injected)
ax2.plot(t_UG75_DX20_x05, Q_mean_UG75_DX20_x05, 'b', label=label_x_equal_5)
ax2.plot(t_UG75_DX20_x10, Q_mean_UG75_DX20_x10, 'k', label=label_x_equal_10)
ax2.plot(t_UG75_DX20_x10, Q_mean_UG75_DX20_x15, 'r', label=label_x_equal_15)
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}20$',fontsize=80*FFIG)
ax2.set_title(label_UG75_DX20)
ax2.xaxis.set_ticks([0,3,6,9,12])
ax2.legend(loc='best',fontsize=40*FFIG)

ax3.plot([t_min, max(t_UG100_DX10_x05)], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
ax3.plot(t_UG100_DX10_x05, Q_mean_UG100_DX10_x05, 'b', label=label_x_equal_5)
ax3.plot(t_UG100_DX10_x10, Q_mean_UG100_DX10_x10, 'k', label=label_x_equal_10)
#ax3.text(0.0,6000,r'$\mathrm{UG}100\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(label_UG100_DX10)
ax3.xaxis.set_ticks([0,0.3,0.6,0.9,1.2])

ax4.plot([t_min, max(t_UG100_DX20_x05)], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
ax4.plot(t_UG100_DX20_x05, Q_mean_UG100_DX20_x05, 'b', label=label_x_equal_5)
ax4.plot(t_UG100_DX20_x10, Q_mean_UG100_DX20_x10, 'k', label=label_x_equal_10)
ax4.plot(t_UG100_DX20_x10, Q_mean_UG100_DX20_x15, 'r', label=label_x_equal_15)
ax4.set_title(label_UG100_DX20)
ax4.xaxis.set_ticks([0,4,8,12,16,20])

axs.flat[0].set(ylabel = y_label_Ql_mean_perp)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,5000)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_mean_Q_iso_x.pdf')
plt.show()
plt.close

#%% RMS values evolution in iso-x planes

fig = plt.figure(figsize=figsize_4_in_a_row)
gs = fig.add_gridspec(1, 4, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3, ax4) = gs.subplots(sharey='row')

ax1.plot(t_UG75_DX10_x05, Q_RMS_UG75_DX10_x05, 'b', label=label_x_equal_5)
ax1.plot(t_UG75_DX10_x10, Q_RMS_UG75_DX10_x10, 'k', label=label_x_equal_10)
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(label_UG75_DX10)
ax1.xaxis.set_ticks([0,1,2,3])
ax1.yaxis.set_ticks([0,500,1000,1500,2000,2500,3000])

ax2.plot(t_UG75_DX20_x05, Q_RMS_UG75_DX20_x05, 'b', label=label_x_equal_5)
ax2.plot(t_UG75_DX20_x10, Q_RMS_UG75_DX20_x10, 'k', label=label_x_equal_10)
ax2.plot(t_UG75_DX20_x10, Q_RMS_UG75_DX20_x15, 'r', label=label_x_equal_15)
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}20$',fontsize=80*FFIG)
ax2.set_title(label_UG75_DX20)
ax2.xaxis.set_ticks([0,3,6,9,12])

ax3.plot(t_UG100_DX10_x05, Q_RMS_UG100_DX10_x05, 'b', label=label_x_equal_5)
ax3.plot(t_UG100_DX10_x10, Q_RMS_UG100_DX10_x10, 'k', label=label_x_equal_10)
#ax3.text(0.0,6000,r'$\mathrm{UG}100\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(label_UG100_DX10)
ax3.xaxis.set_ticks([0,0.3,0.6,0.9,1.2])

ax4.plot(t_UG100_DX20_x05, Q_RMS_UG100_DX20_x05, 'b', label=label_x_less_5)
ax4.plot(t_UG100_DX20_x10, Q_RMS_UG100_DX20_x10, 'k', label=label_x_less_10)
ax4.plot(t_UG100_DX20_x10, Q_RMS_UG100_DX20_x15, 'r', label=label_x_less_15)
#ax4.text(0.5,6000,r'$\mathrm{UG}100\_\mathrm{DX}20$',fontsize=80*FFIG)
ax4.set_title(label_UG100_DX20)
#ax4.yaxis.set_ticks([0,1000,2000,3000])
ax4.xaxis.set_ticks([0,4,8,12,16,20])

axs.flat[0].set(ylabel = y_label_Ql_RMS_perp)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,3000)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_rms_Q_iso_x.pdf')
plt.show()
plt.close



#%% Mean values evolution in filming planes

fig = plt.figure(figsize=figsize_4_in_a_row)
gs = fig.add_gridspec(1, 4, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3, ax4) = gs.subplots(sharey='row')

ax1.plot(t_UG75_DX10_x05_filming, Q_mean_UG75_DX10_x05_filming, 'b', label=label_x_equal_5)
ax1.plot(t_UG75_DX10_x10_filming, Q_mean_UG75_DX10_x10_filming, 'k', label=label_x_equal_10)
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(label_UG75_DX10)
ax1.xaxis.set_ticks([0,1,2,3])
ax1.yaxis.set_ticks([0,50,100,150,200,250,300])

ax2.plot(t_UG75_DX20_x05_filming, Q_mean_UG75_DX20_x05_filming, 'b', label=label_x_less_5)
ax2.plot(t_UG75_DX20_x10_filming, Q_mean_UG75_DX20_x10_filming, 'k', label=label_x_less_10)
ax2.plot(t_UG75_DX20_x15_filming, Q_mean_UG75_DX20_x15_filming, 'r', label=label_x_less_15)
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}20$',fontsize=80*FFIG)
ax2.set_title(label_UG75_DX20)
ax2.xaxis.set_ticks([0,3,6,9,12])
ax2.legend(loc='best',fontsize=40*FFIG)

ax3.plot(t_UG100_DX10_x05_filming, Q_mean_UG100_DX10_x05_filming, 'b', label=label_x_equal_5)
ax3.plot(t_UG100_DX10_x10_filming, Q_mean_UG100_DX10_x10_filming, 'k', label=label_x_equal_10)
#ax3.text(0.0,6000,r'$\mathrm{UG}100\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(label_UG100_DX10)
ax3.xaxis.set_ticks([0,0.3,0.6,0.9,1.2])

ax4.plot(t_UG100_DX20_x05_filming, Q_mean_UG100_DX20_x05_filming, 'b', label=label_x_equal_5)
ax4.plot(t_UG100_DX20_x10_filming, Q_mean_UG100_DX20_x10_filming, 'k', label=label_x_equal_10)
ax4.plot(t_UG100_DX20_x10_filming, Q_mean_UG100_DX20_x15_filming, 'r', label=label_x_equal_15)
#ax4.text(0.5,6000,r'$\mathrm{UG}100\_\mathrm{DX}20$',fontsize=80*FFIG)
ax4.set_title(label_UG100_DX20)
ax4.xaxis.set_ticks([0,4,8,12,16,20])


axs.flat[0].set(ylabel = y_label_Ql_mean_film)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,300)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_mean_Q_filming.pdf')
plt.show()
plt.close








#%% RMS values evolution in filming planes


fig = plt.figure(figsize=figsize_4_in_a_row)
gs = fig.add_gridspec(1, 4, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3, ax4) = gs.subplots(sharey='row')


ax1.plot(t_UG75_DX10_x05_filming, Q_RMS_UG75_DX10_x05_filming, 'b', label=label_x_less_5)
ax1.plot(t_UG75_DX10_x10_filming, Q_RMS_UG75_DX10_x10_filming, 'k', label=label_x_less_10)
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(label_UG75_DX10)
ax1.xaxis.set_ticks([0,1,2,3])
ax1.yaxis.set_ticks([0,40,80,120,160])

ax2.plot(t_UG75_DX20_x05_filming, Q_RMS_UG75_DX20_x05_filming, 'b', label=label_x_less_5)
ax2.plot(t_UG75_DX20_x10_filming, Q_RMS_UG75_DX20_x10_filming, 'k', label=label_x_less_10)
ax2.plot(t_UG75_DX20_x15_filming, Q_RMS_UG75_DX20_x15_filming, 'r', label=label_x_less_15)
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}20$',fontsize=80*FFIG)
ax2.set_title(label_UG75_DX20)
ax2.xaxis.set_ticks([0,3,6,9,12])

ax3.plot(t_UG100_DX10_x05_filming, Q_RMS_UG100_DX10_x05_filming, 'b', label=label_x_less_5)
ax3.plot(t_UG100_DX10_x10_filming, Q_RMS_UG100_DX10_x10_filming, 'k', label=label_x_less_10)
#ax3.text(0.0,6000,r'$\mathrm{UG}100\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(label_UG100_DX10)
ax3.xaxis.set_ticks([0,0.3,0.6,0.9,1.2])

ax4.plot(t_UG100_DX20_x05_filming, Q_RMS_UG100_DX20_x05_filming, 'b', label=label_x_less_5)
ax4.plot(t_UG100_DX20_x10_filming, Q_RMS_UG100_DX20_x10_filming, 'k', label=label_x_less_10)
ax4.plot(t_UG100_DX20_x10_filming, Q_RMS_UG100_DX20_x15_filming, 'r', label=label_x_less_15)
#ax4.text(0.5,6000,r'$\mathrm{UG}100\_\mathrm{DX}20$',fontsize=80*FFIG)
ax4.set_title(label_UG100_DX20)
#ax4.yaxis.set_ticks([0,50,100,150])
ax4.xaxis.set_ticks([0,4,8,12,16,20])


axs.flat[0].set(ylabel = y_label_Ql_RMS_film)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=x_label_time)
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
plt.ylim(0,160)
plt.tight_layout()
plt.savefig(folder_manuscript+'evolution_rms_Q_filming.pdf')
plt.show()
plt.close


#%% Bar graphs iso-x

Q_x_mean_x05 = [Q_mean_UG75_DX10_x05[-1],Q_mean_UG75_DX20_x05[-1],
                     Q_mean_UG100_DX10_x05[-1],Q_mean_UG100_DX20_x05[-1]]
Q_x_mean_x10 = [Q_mean_UG75_DX10_x10[-1],Q_mean_UG75_DX20_x10[-1],
                     Q_mean_UG100_DX10_x10[-1],Q_mean_UG100_DX20_x10[-1]]
Q_x_mean_x15 = [Q_mean_UG75_DX20_x15[-1],
                     Q_mean_UG100_DX20_x15[-1]]

Q_x_RMS_x05 = [Q_RMS_UG75_DX10_x05[-1],Q_RMS_UG75_DX20_x05[-1],
                     Q_RMS_UG100_DX10_x05[-1],Q_RMS_UG100_DX20_x05[-1]]
Q_x_RMS_x10 = [Q_RMS_UG75_DX10_x10[-1],Q_RMS_UG75_DX20_x10[-1],
                     Q_RMS_UG100_DX10_x10[-1],Q_RMS_UG100_DX20_x10[-1]]
Q_x_RMS_x15 = [Q_RMS_UG75_DX20_x15[-1],
                    Q_RMS_UG100_DX20_x15[-1]]



# Bar graph with RMS
plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.plot([r1[0]-barWidth*1.5,r2[0]+barWidth*1.5],[Q_inj_UG75]*2, '--k', label=label_Ql_injected,linewidth=4*FFIG)
plt.plot([r1[2]-barWidth*1.5,r2[-1]+barWidth*1.5],[Q_inj_UG100]*2, '--k',linewidth=4*FFIG)
plt.bar(r1-0.25, Q_x_mean_x05, yerr=Q_x_RMS_x05, width=barWidth, color='blue', edgecolor='white', label=label_x_equal_5, capsize=barWidth*20)
plt.bar(r1, Q_x_mean_x10, yerr=Q_x_RMS_x10, width=barWidth, color='grey', edgecolor='white', label=label_x_equal_10, capsize=barWidth*20)
plt.bar(r2+0.25, Q_x_mean_x15, yerr=Q_x_RMS_x15, width=barWidth, color='red', edgecolor='white', label=label_x_equal_15, capsize=barWidth*20)
#plt.xlabel('Case')#, fontweight='bold')
plt.ylabel(y_label_Ql_mean_perp)
plt.xticks([r for r in range(len(cases))], cases)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(folder_manuscript+'bar_graph_isox_IBs.pdf')
plt.show()
plt.close()

#%% Bar graphs filming


Q_x_film_mean_x05 = [Q_mean_UG75_DX10_x05_filming[-1],Q_mean_UG75_DX20_x05_filming[-1],
                     Q_mean_UG100_DX10_x05_filming[-1],Q_mean_UG100_DX20_x05_filming[-1]]
Q_x_film_mean_x10 = [Q_mean_UG75_DX10_x10_filming[-1],Q_mean_UG75_DX20_x10_filming[-1],
                     Q_mean_UG100_DX10_x10_filming[-1],Q_mean_UG100_DX20_x10_filming[-1]]
Q_x_film_mean_x15 = [Q_mean_UG75_DX20_x15_filming[-1],
                     Q_mean_UG100_DX20_x15_filming[-1]]

Q_x_film_RMS_x05 = [Q_RMS_UG75_DX10_x05_filming[-1],Q_RMS_UG75_DX20_x05_filming[-1],
                     Q_RMS_UG100_DX10_x05_filming[-1],Q_RMS_UG100_DX20_x05_filming[-1]]
Q_x_film_RMS_x10 = [Q_RMS_UG75_DX10_x10_filming[-1],Q_RMS_UG75_DX20_x10_filming[-1],
                     Q_RMS_UG100_DX10_x10_filming[-1],Q_RMS_UG100_DX20_x10_filming[-1]]
Q_x_film_RMS_x15 = [Q_RMS_UG75_DX20_x15_filming[-1],
                    Q_RMS_UG100_DX20_x15_filming[-1]]



# Bar graph with RMS
plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.bar(r1-0.25, Q_x_film_mean_x05, yerr=Q_x_film_RMS_x05, width=barWidth, color='blue', edgecolor='white', label=label_x_less_5, capsize=barWidth*20)
plt.bar(r1, Q_x_film_mean_x10, yerr=Q_x_film_RMS_x10, width=barWidth, color='grey', edgecolor='white', label=label_x_less_10, capsize=barWidth*20)
plt.bar(r2+0.25, Q_x_film_mean_x15, yerr=Q_x_film_RMS_x15, width=barWidth, color='red', edgecolor='white', label=label_x_less_15, capsize=barWidth*20)
#plt.xlabel('Case')#, fontweight='bold')
plt.ylabel(y_label_Ql_mean_film)
plt.xticks([r for r in range(len(cases))], cases)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(folder_manuscript+'bar_graph_filming_IBs.pdf')
plt.show()
plt.close()





#%% Total fluid losses calculations

#UG75_DX10
Q_tot_UG75_DX10_x05 = Q_mean_UG75_DX10_x05[-1] + Q_mean_UG75_DX10_x05_filming[-1]
Q_tot_UG75_DX10_x10 = Q_mean_UG75_DX10_x10[-1] + Q_mean_UG75_DX10_x10_filming[-1]

Q_tot_UG75_DX10 = np.array([Q_tot_UG75_DX10_x05, Q_tot_UG75_DX10_x10])

#UG75_DX20
Q_tot_UG75_DX20_x05 = Q_mean_UG75_DX20_x05[-1] + Q_mean_UG75_DX20_x05_filming[-1]
Q_tot_UG75_DX20_x10 = Q_mean_UG75_DX20_x10[-1] + Q_mean_UG75_DX20_x10_filming[-1]
Q_tot_UG75_DX20_x15 = Q_mean_UG75_DX20_x15[-1] + Q_mean_UG75_DX20_x15_filming[-1]

Q_tot_UG75_DX20 = np.array([Q_tot_UG75_DX20_x05, Q_tot_UG75_DX20_x10, Q_tot_UG75_DX20_x15])

#UG100_DX10
Q_tot_UG100_DX10_x05 = Q_mean_UG100_DX10_x05[-1] + Q_mean_UG100_DX10_x05_filming[-1]
Q_tot_UG100_DX10_x10 = Q_mean_UG100_DX10_x10[-1] + Q_mean_UG100_DX10_x10_filming[-1]

Q_tot_UG100_DX10 = np.array([Q_tot_UG100_DX10_x05, Q_tot_UG100_DX10_x10])

#UG100_DX20
Q_tot_UG100_DX20_x05 = Q_mean_UG100_DX20_x05[-1] + Q_mean_UG100_DX20_x05_filming[-1]
Q_tot_UG100_DX20_x10 = Q_mean_UG100_DX20_x10[-1] + Q_mean_UG100_DX20_x10_filming[-1]
Q_tot_UG100_DX20_x15 = Q_mean_UG100_DX20_x15[-1] + Q_mean_UG100_DX20_x15_filming[-1]

Q_tot_UG100_DX20 = np.array([Q_tot_UG100_DX20_x05, Q_tot_UG100_DX20_x10, Q_tot_UG100_DX20_x15])


# Losses
'''
Q_loss_UG75_DX10  = Q_inj_UG75 - Q_tot_UG75_DX10
Q_loss_UG75_DX20  = Q_inj_UG75 - Q_tot_UG75_DX20
Q_loss_UG100_DX10 = Q_inj_UG100 - Q_tot_UG100_DX10
Q_loss_UG100_DX20 = Q_inj_UG100 - Q_tot_UG100_DX20
'''

Q_loss_UG75_DX10  = (Q_inj_UG75 - Q_tot_UG75_DX10)/Q_inj_UG75*100
Q_loss_UG75_DX20  = (Q_inj_UG75 - Q_tot_UG75_DX20)/Q_inj_UG75*100
Q_loss_UG100_DX10 = (Q_inj_UG100 - Q_tot_UG100_DX10)/Q_inj_UG100*100
Q_loss_UG100_DX20 = (Q_inj_UG100 - Q_tot_UG100_DX20)/Q_inj_UG100*100

#%% Total fluid losses plots

x_dx20 = [5,10,15]
x_dx10 = [5,10]

#Plot
plt.figure(figsize=(FFIG*26,FFIG*16))
plt.plot(x_dx10, Q_loss_UG75_DX10, 'o-k',label=label_UG75_DX10)
plt.plot(x_dx20, Q_loss_UG75_DX20, '^-k',label=label_UG75_DX20)
plt.plot(x_dx10, Q_loss_UG100_DX10, 'o-b',label=label_UG100_DX10)
plt.plot(x_dx20, Q_loss_UG100_DX20, '^-b',label=label_UG100_DX20)
plt.xlabel(r'$x~[\mathrm{mm]}$')
plt.ylabel(r'$\Delta Q_l~[\%]$')
plt.legend(loc='best')
plt.xticks([5,10,15])
plt.yticks([0,10,20,30])
plt.grid()
plt.tight_layout()

plt.savefig(folder_manuscript+'Ql_loss_with_x.pdf')
plt.show()
plt.close()


