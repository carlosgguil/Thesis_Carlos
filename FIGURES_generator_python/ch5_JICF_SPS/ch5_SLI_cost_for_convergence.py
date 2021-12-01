# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:09:07 2020

@author: d601630
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SLI_cost_for_convergence/'
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
# properties of arrows and cotas
linewidth_cotas = 5*FFIG
linewidth_arrow = 5*FFIG
head_width_ = 0.03
head_length_ = 0.175


# Define labels and tags
y_label_t_CPU = r'$t_\mathrm{CPU}~[\cdot 10^6 ~h]$'
y_label_t_CPU_over_t_phys =  r'$t_\mathrm{CPU}/t_\mathrm{ph}~[\cdot 10^6 ~h ~\mathrm{ms}^{-1}]$'

label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
cases = [label_UG75_DX10 , label_UG75_DX20,
         label_UG100_DX10, label_UG100_DX20]

label_UG100_DX20_saving = r'$\mathrm{UG}100\_\mathrm{DX}20$\\\\ ~~~~~~~~~~$t_\mathrm{ph} = X.X~\mathrm{ms} $'
label_UG100_DX10_saving = r'$\mathrm{UG}100\_\mathrm{DX}10$\\\\ ~~~~~~~~~~$t_\mathrm{ph} = X.X~\mathrm{ms} $'
label_UG100_DX10_saving_long = r'$\mathrm{UG}100\_\mathrm{DX}10$\\\\ ~~~~~~~~~~$t_\mathrm{ph} = X.X~\mathrm{ms} $'
cases_saving = [label_UG100_DX20_saving, label_UG100_DX10_saving, label_UG100_DX10_saving_long]



# For bar graphs
barWidth = 0.25
r1 = np.arange(len(cases))
r1_saving = np.arange(len(cases_saving))

# computed, absolute costs in hours (million)
t_cpu  = np.array([0.66, 0.264, 0.66, 0.264])
t_phys = np.array([0.55, 2.5, 0.55, 2.5])
t_cpu_over_t_phys = t_cpu/t_phys

# saving costs for UG100
t_cpu_saving = np.array([t_cpu[3], t_cpu[2], t_cpu_over_t_phys[2]*t_phys[3] ])


#%% Bar graph(s) for computed hours

plt.figure(figsize=figsize_bar)
ax = plt.gca()
ax2 = ax.twinx()
ax.bar(r1-barWidth/2, t_cpu, width=barWidth, color='black', edgecolor='white', capsize=barWidth*20)
ax2.bar(r1+barWidth/2, t_cpu_over_t_phys, width=barWidth, color='blue', edgecolor='white', capsize=barWidth*20)
ax.set_ylabel(y_label_t_CPU)
ax2.set_ylabel(y_label_t_CPU_over_t_phys)
ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
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
plt.text(x_arrow-0.08, y_arrow-l_arrow/2, r'$\mathrm{Cost~\\savings}$',
             color='black', rotation='vertical',fontsize=90*FFIG)

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

