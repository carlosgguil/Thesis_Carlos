# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 60*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.markersize'] = 40*FFIG

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/'


#%% Load sprays

# Parameters of simulations
params_simulation_UG100 = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                           'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                           'SIGMA': 22e-3,
                           'D_inj': 0.45e-3}

params_simulation_UG75 = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 17.5,
                          'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 75,
                          'SIGMA': 22e-3,
                          'D_inj': 0.45e-3}
params_simulation_UG100['Q_inj'] = np.pi/4*params_simulation_UG100['D_inj']**2*params_simulation_UG100['U_L']
params_simulation_UG75['Q_inj'] = np.pi/4*params_simulation_UG75['D_inj']**2*params_simulation_UG75['U_L']

# Load sprays
sp1, sp2, sp3, sp4, sp5 = load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100)
sprays_list_UG75_DX10  = sp1
sprays_list_UG75_DX20  = sp2
sprays_list_UG100_DX10 = sp3
sprays_list_UG100_DX20 = sp4
sprays_list_UG100_DX20_NT = sp5

sprays_list_all = [sp1, sp2, sp3, sp4, sp5]

#%% Get SMD
SMD_cases = []
for i in range(len(sprays_list_all)):
    case = sprays_list_all[i]
    SMD_val = []
    for j in range(len(case)):
        SMD_val.append(case[j].SMD)
    SMD_cases.append(SMD_val)
        
    

#%% parameters and plot

# legend labels
label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_UG100_DX20_NT = r'$\mathrm{UG100}\_\mathrm{DX20}\_\mathrm{NT}$'
labels_ = [label_UG75_DX10 , label_UG75_DX20,
           label_UG100_DX10, label_UG100_DX20,
           label_UG100_DX20_NT]

# axis labels
x_label_x = r'$x~[\mathrm{mm}]$'
y_label_SMD   = r'$\mathrm{SMD}~[\mu \mathrm{m}]$'

x_DX20 = [5,10,15]
x_DX10 = [5,10]

SMD_UG75_DX20  = [139.2, 130.7, 124.8, 121.5]
SMD_UG75_DX10  = [80.5, 69.4]

SMD_UG100_DX20 = [132.7, 120.7, 117.7, 118.6]
SMD_UG100_DX10 = [72.0, 64.6]




#%% Plot all in one graph
plt.figure(figsize=(FFIG*22,FFIG*13))
i = 0; plt.plot(x_DX10, SMD_cases[i], '*-k',label=labels_[i])
i = 1; plt.plot(x_DX20, SMD_cases[i], '^-k',label=labels_[i])
i = 2; plt.plot(x_DX10, SMD_cases[i], '*-b',label=labels_[i])
i = 3; plt.plot(x_DX20, SMD_cases[i], '^-b',label=labels_[i])
i = 4; plt.plot(x_DX10, SMD_cases[i], '^-r',label=labels_[i])
plt.xlabel(x_label_x)
plt.ylabel(y_label_SMD )
plt.legend(loc='best')
plt.xticks([5,10,15])
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_values.pdf')
plt.show()
plt.close()

#%% Plot with broken y axis

f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(FFIG*22,FFIG*15))


# Coarse cases
i = 1; ax.plot(x_DX20, SMD_cases[i], '^-k',label=labels_[i])
i = 3; ax.plot(x_DX20, SMD_cases[i], '^-b',label=labels_[i])
i = 4; ax.plot(x_DX10, SMD_cases[i], '^-r',label=labels_[i])

# Fine cases
i = 0; ax2.plot(x_DX10, SMD_cases[i], '*-k',label=labels_[i]) # UG75_DX10
i = 2; ax2.plot(x_DX10, SMD_cases[i], '*-b',label=labels_[i]) # UG100_DX10

ax.set_ylim(139, 170) # coarse cases
ax.set_yticks([140,150,160,170])
ax2.set_ylim(79, 90) # fine cases
ax2.set_yticks([80,85,90])

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# plot diagonal lines
diagonal_linewidth = 2
d = .010  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), linewidth=diagonal_linewidth, **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), linewidth=diagonal_linewidth,**kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), linewidth=diagonal_linewidth,**kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), linewidth=diagonal_linewidth,**kwargs)  # bottom-right diagonal

ax.grid()
ax2.grid()
ax2.set_xlabel(x_label_x)
ax2.set_xticks([5,10,15])

#ax.set_ylabel(y_label_SMD )
ax.set_title(y_label_SMD)

plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_values_broken_axis.pdf')
plt.show()
plt.close()
