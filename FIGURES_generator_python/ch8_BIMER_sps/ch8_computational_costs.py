# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SLI_cost_for_convergence/'


# Change size of figures if wished
FFIG = 0.5
figsize_ = (FFIG*30,FFIG*20)
figsize_4_in_a_row = (FFIG*55,FFIG*15)
figsize_bar = (FFIG*30,FFIG*20)

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 120*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 120*FFIG # 80*FFIG
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
# properties of arrows and cotas
linewidth_cotas = 5*FFIG
linewidth_arrow = 5*FFIG
head_width_ = 0.03
head_length_ = 0.175


# Define labels and tags
y_label_t_CPU = r'$t_\mathrm{CPU}~[\cdot 10^5 ~h]$'
y_label_t_CPU_over_t_phys =  r'$t_\mathrm{CPU}/t_\mathrm{ph}~[\cdot 10^5 ~h.\mathrm{ms}^{-1}]$'

label_DX10  = r'$\mathrm{DX}10$'
label_DX15  = r'$\mathrm{DX}15$'
cases = [label_DX10 , label_DX15]



label_DX15_saving = r'$~~~~\mathrm{DX}15$'
label_DX10_saving = r'$~~~~\mathrm{DX}10$'
label_DX10_saving_long = r'$~~~~\mathrm{DX}10$'


#label_DX15_saving = r'$~~~~\mathrm{DX}15$\\\\ ~~~~~~~~~~ $t_\mathrm{acc} = 2.64~\mathrm{ms} $'
#label_DX10_saving = r'$~~~~\mathrm{DX}10$\\\\ ~~~~~~~~~~$t_\mathrm{acc} = 0.96~\mathrm{ms} $'
#label_DX10_saving_long = r'$~~~~\mathrm{DX}10$\\\\ ~~~~~~~~~~$t_\mathrm{acc} = 2.64~\mathrm{ms} $'
cases_saving = [label_DX15_saving, label_DX10_saving, label_DX10_saving_long]



# For bar graphs
barWidth = 0.25
r1 = np.arange(len(cases))
r1_saving = np.arange(len(cases_saving))

# computed, absolute costs in hours (10e5)
t_cpu  = np.array([6.976, 4.4736])
t_phys = np.array([1.7303, 3.8078])
#t_ = np.array([1.0756, 6.2961, 0.7826, 6.1595, 6.2894])
t_cpu_over_t_phys = t_cpu/t_phys

# saving costs for BIMER
t_cpu_saving = np.array([t_cpu[1], t_cpu[0], t_cpu_over_t_phys[0]*t_phys[1] ])


#%% Bar graph(s) for computed hours

plt.figure(figsize=(FFIG*30,FFIG*15))
ax = plt.gca()
ax2 = ax.twinx()
ax.bar(r1-barWidth/2, t_cpu, width=barWidth, color='black', edgecolor='white', capsize=barWidth*20)
ax2.bar(r1+barWidth/2, t_cpu_over_t_phys, width=barWidth, color='grey', edgecolor='white', capsize=barWidth*20)
ax.set_ylabel(y_label_t_CPU)
ax2.set_ylabel(y_label_t_CPU_over_t_phys)
ax2.tick_params(axis='y', colors='grey')
ax2.yaxis.label.set_color('grey')
plt.xticks([r for r in range(len(cases))], cases)
plt.tight_layout()
plt.savefig(folder_manuscript+'cost_all_simulations.pdf')
plt.show()
plt.close()


#%% Bar graph(s) for estimated hours (cost for convergence)
x_arrow = r1_saving[1]+barWidth*2-0.05
y_arrow = (t_cpu_saving[2]+t_cpu_saving[1])/2
l_arrow = (t_cpu_saving[2]-t_cpu_saving[1])/2

plt.figure(figsize=figsize_bar)
plt.bar(r1_saving, t_cpu_saving, width=barWidth, color='black', edgecolor='white', capsize=barWidth*20)
# Plot lines and arrows
plt.plot([r1_saving[1]+barWidth/2+0.02,r1_saving[1]+barWidth*2],  
         [t_cpu_saving[1]]*2, '--k', linewidth = linewidth_cotas)
plt.plot([r1_saving[1]+barWidth*2-0.1, r1_saving[2]-barWidth/2-0.02],  
         [t_cpu_saving[2]]*2, '--k', linewidth = linewidth_cotas)
plt.arrow(x_arrow, y_arrow, 0, l_arrow, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
plt.arrow(x_arrow, y_arrow, 0, -1*l_arrow, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
plt.text(x_arrow-0.15, y_arrow-l_arrow/1.5, r'$\mathrm{Cost~\\savings}$',
             color='black', rotation='vertical',fontsize=90*FFIG)
# texts with accumulation times
plt.text(-0.40, -2.5, r'$t_\mathrm{acc} = 2.64~\mathrm{ms} $',
         color='black', fontsize=90*FFIG)
plt.text(0.60, -2.5, r'$t_\mathrm{acc} = 0.96~\mathrm{ms} $',
         color='black', fontsize=90*FFIG)
plt.text(1.60, -2.5, r'$t_\mathrm{acc} = 2.64~\mathrm{ms} $',
         color='black', fontsize=90*FFIG)

'''    
    # Text
    plt.text(t_arrow_NF_loss+0.01, vol_ls_phi_integrated_val[-1]+dv_NF/3, r'$\Delta V_\mathrm{NF}$',
             color='red', rotation='vertical',fontsize=50*FFIG)
    plt.text(t_arrow_total_loss+0.01, vol_ls_phi_integrated_val[-1]+dv_NF/3, r'$\Delta V_\mathrm{Total}$',
             color='black', rotation='vertical',fontsize=50*FFIG)
'''       
plt.ylabel(y_label_t_CPU)
plt.xticks([r for r in range(len(cases_saving))], cases_saving)
plt.tight_layout()
plt.savefig(folder_manuscript+'cost_savings_simulations.pdf')
plt.show()
plt.close()

