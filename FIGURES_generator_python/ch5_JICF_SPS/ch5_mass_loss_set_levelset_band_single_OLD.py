# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FFIG = 0.5
SCALE_FACTOR = 1e9
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/flow_rates_mass_loss_set_levelset_band/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/mass_loss_due_to_levelset/jicf_data_processing/'





#%% Parameters

tau_in_UG100 =  19.29e-3

x_label_time = r'$t^{*}$'
y_label_volume = r'$V_l~[\mathrm{mm}^3]$'

label_ls_volume  = r'$V_l$'
label_SLB_volume  = r'$V_{l,\mathrm{NF}}$'
label_injected_volume  = r'$V_{l,\mathrm{inj}}$'


#%% Stuff

case  = 'dx20_NSTEPS'



PLOT_ADAPTATION_ITERS = True
dt = 0.011 # In ms

if case == 'dx20_baseline':
    folder =   folder+'ls_phi_quantification_dx20_baseline_restart01_6heures'
elif case == 'dx20_EPSILON_01':
    folder =   folder+'ls_phi_quantification_dx20_LS_EPSILON0p7_restart01_12heures'
elif case == 'dx20_EPSILON_02':
    folder =   folder+'ls_phi_quantification_dx20_LS_EPSILON0p7_restart02'
elif case == 'dx10':
    folder =   folder+'ls_phi_quantification_dx10_restart_18_24heures'
elif case ==   'dx20_NSTEPS':
    folder = folder+'ls_phi_quantification_dx20_LS_NSTEPS6_restart01_12heures'
    dt = None


# Get data from dataframes
df = pd.read_csv(folder+'/data_to_plot.csv')

time       = df['time'].values
vol_ls_phi_integrated   = df['vol_ls_phi_integrated'].values
vol_ls_phi_int_plus_SLB = df['vol_ls_phi_int_plus_SLB'].values
V_injected = df['V_injected'].values
adapt_time = df[~df['adapt_time'].isna()]['adapt_time'].values

#%% Plot

time_plot = time/tau_in_UG100
adapt_time_plot = adapt_time/tau_in_UG100

# Plot as function of time
plt.figure(figsize=figsize_)
axes = plt.gca()
plt.plot(time_plot, vol_ls_phi_integrated, 'k', label=label_ls_volume)
plt.plot(time_plot, vol_ls_phi_int_plus_SLB, 'r', label=label_SLB_volume)
plt.plot(time_plot, V_injected, 'b', label=label_injected_volume)
ylim  = axes.get_ylim()
if PLOT_ADAPTATION_ITERS:
    for i in range(len(adapt_time_plot)):
        plt.plot([adapt_time_plot[i]]*2,ylim,'--k',alpha=0.3,zorder=0)
else: plt.grid()
plt.xlabel(x_label_time)
plt.ylabel(y_label_volume)
plt.xlim((-0.5e-3 + time_plot[0],time_plot[-1]+0.25e-3))
#plt.xlim(x_lim)
plt.ylim(ylim)
plt.legend(framealpha=1.0, loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'vl_loss_case_'+case+'.pdf')
plt.show()
plt.close

