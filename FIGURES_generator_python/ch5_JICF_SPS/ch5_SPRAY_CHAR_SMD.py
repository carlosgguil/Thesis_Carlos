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




#Plot
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


plt.rcParams['text.usetex'] = False