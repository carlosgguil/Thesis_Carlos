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
from functions import calculate_mean_and_rms, time_average

# Change size of figures if wished
FFIG = 0.5
figsize_ = (FFIG*30,FFIG*20)

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
plt.rcParams['text.usetex'] = True


OP = 'uG100_dx20'





barWidth = 0.50

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
x_label_time  = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
y_label_Ql    = r"$Q_l ~[\mathrm{mm}^3~\mathrm{s}^{-1}]$"

label_Ql_injected = r'$Q_l ~\mathrm{injected}$'

label_x_equal_5  = r'$x = 5~\mathrm{mm}$'
label_x_equal_10 = r'$x = 10~\mathrm{mm}$'
label_x_equal_15 = r'$x = 15~\mathrm{mm}$'

label_x_less_5  = r'$x < 5~\mathrm{mm}$'
label_x_less_10 = r'$x < 10~\mathrm{mm}$'
label_x_less_15 = r'$x < 15~\mathrm{mm}$'


#%% Read iso-x dataframes
df_UG100_DX10_x05 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_x05')
df_UG100_DX10_x10 = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_x10')

# Read filming dataframes
df_UG100_DX10_x05_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_film_x05')
df_UG100_DX10_x10_filming = pd.read_csv(folder+'/overall_integrated_fluxes/uG100_dx10_Q_film_x10')

#%% Extract time and Qs

t_UG100_DX10_x05 = df_UG100_DX10_x05['t_x05'].values
t_UG100_DX10_x05 = (t_UG100_DX10_x05-t_UG100_DX10_x05[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x05 = df_UG100_DX10_x05['Q_t_x05'].values

t_UG100_DX10_x05_filming = df_UG100_DX10_x05_filming['t_film_x05'].values
t_UG100_DX10_x05_filming = (t_UG100_DX10_x05_filming-t_UG100_DX10_x05_filming[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x05_filming = df_UG100_DX10_x05_filming['Q_t_film_x05'].values*SCALE_FACTOR

t_UG100_DX10_x10 = df_UG100_DX10_x10['t_x10'].values
t_UG100_DX10_x10 = (t_UG100_DX10_x10-t_UG100_DX10_x10[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x10 = df_UG100_DX10_x10['Q_t_x10'].values

t_UG100_DX10_x10_filming = df_UG100_DX10_x10_filming['t_film_x10'].values
t_UG100_DX10_x10_filming = (t_UG100_DX10_x10_filming-t_UG100_DX10_x10_filming[0])/tau_ph_UG100_DX10
Q_inst_UG100_DX10_x10_filming = df_UG100_DX10_x10_filming['Q_t_film_x10'].values*SCALE_FACTOR


#%% Plot time evolution of instantaneous Qs for UG100_DX10

t_min = min(t_UG100_DX10_x05)
t_max = max(t_UG100_DX10_x05)

# Iso-x Qs
plt.figure(figsize=figsize_)
#plt.title(r"$Q_l$ evolution ")
plt.plot(t_UG100_DX10_x05, Q_inst_UG100_DX10_x05, 'b', label=label_x_equal_5)
plt.plot(t_UG100_DX10_x10, Q_inst_UG100_DX10_x10, 'k', label=label_x_equal_10)
plt.plot([t_min, t_max], [Q_inj_UG100]*2, '--k', label=label_Ql_injected)
plt.xlabel(x_label_time)
plt.ylabel(y_label_Ql)
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
plt.ylabel(y_label_Ql)
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'inst_Q_iso_x_UG100_dx10_filming.pdf')
plt.show()
plt.close

#%% Plot mean evolution of instantaneous Qs for UG100_DX10
