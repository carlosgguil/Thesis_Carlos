# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('../..')
from sli_functions import load_all_BIMER_global_sprays

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 60*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.markersize'] = 40*FFIG

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/'


#%% Load sprays

# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}

sampling_planes = ['xD_05p00','xD_06p67',
                   'xD_08p33','xD_10p00']    

# Load sprays
sp1, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation, sampling_planes = sampling_planes)

sprays_list_all = [sp3, sp2, sp1]


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
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07 = r'$\mathrm{DX}07$'
labels_ = [label_DX15 , label_DX10,label_DX07]

# axis labels
x_label_x = r'$x_c ~\left[ \mathrm{mm} \right]$'
y_label_SMD   = r'$\mathrm{SMD}~[\mu \mathrm{m}]$'

xD = np.array([5,6.67, 8.33, 10])
xD = xD*0.3



#%% Plot all in one graph
plt.figure(figsize=(FFIG*22,FFIG*13))
i = 0; plt.plot(xD, SMD_cases[i], '^-k',label=labels_[i])
i = 1; plt.plot(xD, SMD_cases[i], '^-b',label=labels_[i])
#i = 2; plt.plot(xD, SMD_cases[i], '^-r',label=labels_[i])
plt.xlabel(x_label_x)
plt.ylabel(y_label_SMD )
plt.legend(loc='best')
#plt.xticks([5, 6, 7, 8, 9, 10])
plt.xticks([1.5, 2, 2.5, 3])
#plt.yticks([35, 40, 45, 50, 55])
plt.ylim(16,60)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_values.pdf')
plt.show()
plt.close()
