"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

   



import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays




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

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/velocities/'

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

sprays_list_all = [sp1, sp2, sp3, sp4, sp5]

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

# legend labels
label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_UG100_DX20_NT = r'$\mathrm{UG100}\_\mathrm{DX20}\_\mathrm{NT}$'
labels_OP = [label_UG75_DX10 , label_UG75_DX20,
                label_UG100_DX10, label_UG100_DX20,
                label_UG100_DX20_NT]



label_x05 = r'$x = 5~\mathrm{mm}$'
label_x10 = r'$x = 10~\mathrm{mm}$'
label_x15 = r'$x = 15~\mathrm{mm}$'
labels_ = [label_x05, label_x10, label_x15]

# x coordinates
x_dx20 = [5, 10, 15]
x_dx10 = [5, 10]


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
        ux_mean_val.append(case[j].u_mean[0])
        ux_rms_val.append(case[j].u_rms[0])
        uy_mean_val.append(case[j].u_mean[1])
        uy_rms_val.append(case[j].u_rms[1])
        uz_mean_val.append(case[j].u_mean[2])
        uz_rms_val.append(case[j].u_rms[2])
        # Volume-weighted
        ux_mean_vw_val.append(case[j].u_mean_volume_weighted[0])
        ux_rms_vw_val.append(case[j].u_rms_volume_weighted[0])
        uy_mean_vw_val.append(case[j].u_mean_volume_weighted[1])
        uy_rms_vw_val.append(case[j].u_rms_volume_weighted[1])
        uz_mean_vw_val.append(case[j].u_mean_volume_weighted[2])
        uz_rms_vw_val.append(case[j].u_rms_volume_weighted[2])
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

#%% UG75 plots
 
# u mean x
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, ux_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, ux_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_mean_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_ux_mean.pdf')
plt.show()
plt.close()

# u rms x
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, ux_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, ux_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_rms_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_ux_rms.pdf')
plt.show()
plt.close()




# u mean z
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, uz_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, uz_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, uz_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, uz_mean_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_uz_mean.pdf')
plt.show()
plt.close()

# u rms z
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, uz_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, uz_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, uz_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, uz_rms_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_uz_rms.pdf')
plt.show()
plt.close()


#%% UG100 plots

# u mean x
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, ux_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_mean_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, ux_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_mean_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, ux_mean_cases[i], '-or', label=labels_OP[i])
ax.plot(x_dx10, ux_mean_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_ux_mean.pdf')
plt.show()
plt.close()

# u rms x
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, ux_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_rms_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, ux_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_rms_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, ux_rms_cases[i], '-or', label=labels_OP[i])
ax.plot(x_dx10, ux_rms_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_ux_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_ux_rms.pdf')
plt.show()
plt.close()


# u mean z
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, ux_mean_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_mean_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, ux_mean_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_mean_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, ux_mean_cases[i], '-or', label=labels_OP[i])
ax.plot(x_dx10, ux_mean_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_uz_mean.pdf')
plt.show()
plt.close()

# u rms z
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, ux_rms_cases[i], '-ok', label=labels_OP[i])
ax.plot(x_dx10, ux_rms_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, ux_rms_cases[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, ux_rms_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, ux_rms_cases[i], '-or', label=labels_OP[i])
ax.plot(x_dx10, ux_rms_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_uz_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_uz_rms.pdf')
plt.show()
plt.close()

