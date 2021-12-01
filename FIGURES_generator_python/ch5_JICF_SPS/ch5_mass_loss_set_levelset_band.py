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
PLOT_ADAPTATION_ITERS = True
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True





figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/flow_rates_mass_loss_set_levelset_band/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/mass_loss_due_to_levelset/jicf_data_processing/data_all_cases/'





#%% Parameters

tau_in_UG100 =  19.29e-3

x_label_time = r'$t^{*}$'
y_label_volume = r'$\mathrm{Liquid}~\mathrm{volume}~[\mathrm{mm}^3]$'

label_ls_volume  = r'$V_l$'
label_SLB_volume  = r'$V_{l,\mathrm{NF}}$'
label_injected_volume  = r'$V_{l,\mathrm{inj}}$'

labels_save = ['baseline', 'epsilon', 'Nsteps', 'dx10'] 



# properties of arrows and cotas
linewidth_cotas = 5*FFIG
linewidth_arrow = 5*FFIG
head_width_ = 0.015
head_length_ = 0.002

#%% Read data

df_baseline = pd.read_csv(folder+'/baseline.csv')
df_epsilon  = pd.read_csv(folder+'/epsilon_01.csv')
df_Nsteps   = pd.read_csv(folder+'/Nsteps_iter.csv')
df_dx10     = pd.read_csv(folder+'/dx10.csv')

dataframes = [df_baseline, df_epsilon, df_Nsteps, df_dx10]

time = []
vol_ls_phi_integrated = []
vol_ls_phi_NF = []
vol_ls_phi_injected = []
adapt_time = []
time_plot = []
adapt_time_plot = []
for i in range(len(dataframes)):
    
    # get dataframe
    df = dataframes[i]
    
    # get data from dataframes
    t_val = df['time'].values
    vol_ls_phi_integrated_val = df['vol_ls_phi_integrated'].values
    vol_ls_phi_NF_val = df['vol_ls_phi_int_plus_SLB'].values
    vol_ls_phi_injected_val = df['V_injected'].values
    adapt_time_val = df[~df['adapt_time'].isna()]['adapt_time'].values
    t_val_plot = t_val/tau_in_UG100
    adapt_time_val_plot = adapt_time_val/tau_in_UG100
    
    # append to array
    time.append(t_val)
    vol_ls_phi_integrated.append(vol_ls_phi_integrated_val)
    vol_ls_phi_NF.append(vol_ls_phi_NF_val)
    vol_ls_phi_injected.append(vol_ls_phi_injected_val)
    adapt_time.append(adapt_time_val)
    time_plot.append(t_val_plot)
    adapt_time_plot.append(adapt_time_val_plot)
    
    dv_total = vol_ls_phi_injected_val[-1] - vol_ls_phi_integrated_val[-1]
    dv_NF    = vol_ls_phi_NF_val[-1] - vol_ls_phi_integrated_val[-1]
    
    # plot graph 
    if i == 0:
        fig = plt.figure(figsize=(FFIG*29.3,FFIG*16))
    else:
        fig = plt.figure(figsize=figsize_)
    axes = plt.gca()
    plt.plot(t_val_plot, vol_ls_phi_integrated_val, 'k', label=label_ls_volume)
    plt.plot(t_val_plot, vol_ls_phi_NF_val, 'r', label=label_SLB_volume)
    plt.plot(t_val_plot, vol_ls_phi_injected_val, 'b', label=label_injected_volume)
    ylim  = axes.get_ylim()
    if PLOT_ADAPTATION_ITERS:
        for i in range(len(adapt_time_plot)):
            plt.plot([adapt_time_plot[i]]*2,ylim,'--k',alpha=0.3,zorder=0)
    else: plt.grid()
    plt.xlabel(x_label_time)
    plt.ylabel(y_label_volume)
    plt.xlim((t_val_plot[0],t_val_plot[-1]+0.01))
    #plt.xlim(x_lim)
    plt.ylim(ylim)
    if i == 0:
        plt.legend(framealpha=1.0, loc='best')
        
        plt.plot( [t_val_plot[-1]+0.005,t_val_plot[-1]+0.1],  [vol_ls_phi_injected_val[-1]]*2,
                 '--k', clip_on = False, linewidth = linewidth_cotas)
        plt.plot( [t_val_plot[-1]+0.005,t_val_plot[-1]+0.1],  [vol_ls_phi_integrated_val[-1]]*2,
                 '--k', clip_on = False, linewidth = linewidth_cotas)
        plt.plot( [t_val_plot[-1]+0.005,t_val_plot[-1]+0.05],  [vol_ls_phi_NF_val[-1]]*2,
                 '--k', clip_on = False, linewidth = linewidth_cotas)

        # Arrows
        t_arrow_total_loss = t_val_plot[-1]+0.075
        plt.arrow(t_arrow_total_loss, vol_ls_phi_integrated_val[-1]+dv_total/2, 0, dv_total/2*0.95, head_width=head_width_, head_length=head_length_, 
                  linewidth=linewidth_arrow, color='k', length_includes_head=True, clip_on = False)
        plt.arrow(t_arrow_total_loss, vol_ls_phi_integrated_val[-1]+dv_total/2, 0, -1*dv_total/2*0.95, head_width=head_width_, head_length=head_length_, 
                  linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
        
        t_arrow_NF_loss = t_val_plot[-1]+0.025
        plt.arrow(t_arrow_NF_loss, vol_ls_phi_integrated_val[-1]+dv_NF/2, 0, dv_total/2*0.85, head_width=head_width_, head_length=head_length_, 
                  linewidth=linewidth_arrow, color='r', length_includes_head=True, clip_on = False)
        plt.arrow(t_arrow_NF_loss, vol_ls_phi_integrated_val[-1]+dv_NF/2, 0, -1*dv_total/2*0.85, head_width=head_width_, head_length=head_length_, 
                  linewidth=linewidth_arrow, color='r', shape = 'full', length_includes_head=True, clip_on = False)
        
        # Text
        plt.text(t_arrow_NF_loss+0.01, vol_ls_phi_integrated_val[-1]+dv_NF/3, r'$\Delta V_\mathrm{NF}$',
                 color='red', rotation='vertical',fontsize=50*FFIG)
        plt.text(t_arrow_total_loss+0.01, vol_ls_phi_integrated_val[-1]+dv_NF/3, r'$\Delta V_\mathrm{Total}$',
                 color='black', rotation='vertical',fontsize=50*FFIG)
        
    plt.tight_layout()
    plt.savefig(folder_manuscript+'vl_loss_case_'+labels_save[i]+'.pdf')
    plt.show()
    plt.close
    


