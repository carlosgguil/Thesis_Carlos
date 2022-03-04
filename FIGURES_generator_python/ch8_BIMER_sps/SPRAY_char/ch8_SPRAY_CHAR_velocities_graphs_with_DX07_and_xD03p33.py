"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

   



import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('../..')
from sli_functions import load_all_BIMER_global_sprays




FFIG = 0.5
SCALE_FACTOR = 1e9
PLOT_ADAPTATION_ITERS = True
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 80*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['lines.markersize'] =  40*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*26,FFIG*16)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/velocities/'

#%% Load sprays


# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    

# Load sprays
sp1, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation)

sprays_list_all = [sp1, sp2, sp3]

#%% Parameters


format_separating_line = 'k'
linewidth_separating_line = 15*FFIG
linewidth_Ql = 6*FFIG

# axis labels
x_label_  = r'$x~[\mathrm{mm}]$' #r'$t~[\mathrm{ms}]$'
y_label_ux_mean = r'$\overline{u}_x~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_ux_rms  = r'$u_{x,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy_mean = r'$\overline{u}_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz_mean = r'$\overline{u}_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy_rms = r'$u_{y,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz_rms = r'$u_{z,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'

# legend labels# legend labels
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07 = r'$\mathrm{DX}07$'
labels_OP = [label_DX07 , label_DX10, label_DX15]


label_xD03p33 = r'$x/d_\mathrm{inj} = 3.33$'
label_xD05p00 = r'$x/d_\mathrm{inj} = 5.00$'
label_xD06p67 = r'$x/d_\mathrm{inj} = 6.67$'
labels_ = [label_xD03p33, label_xD05p00, label_xD06p67]

# x coordinates
xD = [3.33,5,6.67]
x_lim = (3,7)
x_ticks = [3.33,5,6.67]


#%% Get dimensionless time and velocities evolution

ux_mean_cases = []; uy_mean_cases = []; uz_mean_cases = []
ux_rms_cases  = []; uy_rms_cases  = []; uz_rms_cases  = []
ux_mean_vw_cases = []; uy_mean_vw_cases = []; uz_mean_vw_cases = []
ux_rms_vw_cases  = []; uy_rms_vw_cases  = []; uz_rms_vw_cases  = []
for i in range(len(sprays_list_all)):
    case = sprays_list_all[i]
    ux_mean_val = []; uy_mean_val = []; uz_mean_val = []
    ux_rms_val = []; uy_rms_val = []; uz_rms_val = []
    ux_mean_vw_val = []; uy_mean_vw_val = []; uz_mean_vw_val = []
    ux_rms_vw_val = []; uy_rms_vw_val = []; uz_rms_vw_val = []
    for j in range(len(case)):
        # Time-averaged
        ux_mean_val.append(case[j].uc_mean[0])
        ux_rms_val.append(case[j].uc_rms[0])
        uy_mean_val.append(case[j].uc_mean[1])
        uy_rms_val.append(case[j].uc_rms[1])
        uz_mean_val.append(case[j].uc_mean[2])
        uz_rms_val.append(case[j].uc_rms[2])
        # Volume-weighted
        ux_mean_vw_val.append(case[j].uc_mean_volume_weighted[0])
        ux_rms_vw_val.append(case[j].uc_rms_volume_weighted[0])
        uy_mean_vw_val.append(case[j].uc_mean_volume_weighted[1])
        uy_rms_vw_val.append(case[j].uc_rms_volume_weighted[1])
        uz_mean_vw_val.append(case[j].uc_mean_volume_weighted[2])
        uz_rms_vw_val.append(case[j].uc_rms_volume_weighted[2])
    ux_mean_cases.append(ux_mean_val)
    ux_rms_cases.append(ux_rms_val)
    uy_mean_cases.append(uy_mean_val)
    uy_rms_cases.append(uy_rms_val)
    uz_mean_cases.append(uz_mean_val)
    uz_rms_cases.append(uz_rms_val)
    
    ux_mean_vw_cases.append(ux_mean_vw_val)
    ux_rms_vw_cases.append(ux_rms_vw_val)
    uy_mean_vw_cases.append(uy_mean_vw_val)
    uy_rms_vw_cases.append(uy_rms_vw_val)
    uz_mean_vw_cases.append(uz_mean_vw_val)
    uz_rms_vw_cases.append(uz_rms_vw_val)
        


# Plots 

#%% Velocity plots
 
# u mean x
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, ux_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, ux_mean_vw_cases[i], '--ok')
i = 1
ax.plot(xD, ux_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, ux_mean_vw_cases[i], '--ob')
i = 2
ax.plot(xD, ux_mean_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, ux_mean_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ux_mean.pdf')
plt.show()
plt.close()

# u rms x
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, ux_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, ux_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(xD, ux_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, ux_rms_vw_cases[i], '--ob')
i = 2
ax.plot(xD, ux_rms_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, ux_rms_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ux_rms.pdf')
plt.show()
plt.close()

# u mean y
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, uy_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, uy_mean_vw_cases[i], '--ok')
i = 1
ax.plot(xD, uy_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, uy_mean_vw_cases[i], '--ob')
i = 2
ax.plot(xD, uy_mean_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, uy_mean_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'uy_mean.pdf')
plt.show()
plt.close()

# u rms y
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, uy_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, uy_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(xD, uy_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, uy_rms_vw_cases[i], '--ob')
i = 2
ax.plot(xD, uy_rms_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, uy_rms_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'uy_rms.pdf')
plt.show()
plt.close()





# u mean z
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, uz_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, uz_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(xD, uz_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, uz_mean_vw_cases[i], '--ob')
i = 2
ax.plot(xD, uz_mean_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, uz_mean_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'uz_mean.pdf')
plt.show()
plt.close()

# u rms z
plt.figure(figsize=figsize_)
i = 0
ax  = plt.gca()
ax.plot(xD, uz_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(xD, uz_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(xD, uz_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(xD, uz_rms_vw_cases[i], '--ob')
i = 2
ax.plot(xD, uz_rms_cases[i], '-or', label=labels_OP[i])
ax.plot(xD, uz_rms_vw_cases[i], '--or')
ax.set_xlim(x_lim)
ax.set_xticks(x_ticks)
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'uz_rms.pdf')
plt.show()
plt.close()

