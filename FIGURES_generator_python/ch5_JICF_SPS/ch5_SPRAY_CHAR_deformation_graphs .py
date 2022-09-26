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
figsize_double = (FFIG*27.5,FFIG*22.5)


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/deformation/'

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
y_label_alpha_mean = r'$\overline{\alpha}_\mathrm{VW}$'
y_label_alpha_rms  = r'$\alpha_\mathrm{RMS}$'
y_label_beta_mean = r'$\overline{\beta}_\mathrm{VW}$'
y_label_beta_rms = r'$\beta_\mathrm{RMS}$'
y_label_alpha_beta_mean = r'$\hspace{-1in}\overline{\beta}_\mathrm{VW} ~~~~~~~~~~~~~~~~~~~~ \overline{\alpha}_\mathrm{VW}$'



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

alpha_mean_cases = []; beta_mean_cases = []
alpha_rms_cases  = []; beta_rms_cases  = []
alpha_mean_vw_cases = []; beta_mean_vw_cases = []
alpha_rms_vw_cases  = []; beta_rms_vw_cases  = []
for i in range(len(sprays_list_all)):
    case = sprays_list_all[i]
    alpha_mean_val = []; beta_mean_val = []
    alpha_rms_val = []; beta_rms_val = []
    alpha_mean_vw_val = []; beta_mean_vw_val = []
    alpha_rms_vw_val = []; beta_rms_vw_val = []
    for j in range(len(case)):
        spray = case[j] 
        # Time-averaged
        alpha_mean_val.append(spray.alpha.mean)
        alpha_rms_val.append(spray.alpha.std)
        beta_mean_val.append(spray.beta.mean)
        beta_rms_val.append(spray.beta.std)
        # Volume-weighted
        num_alpha = 0; num_beta = 0
        for n in range(spray.n_droplets):
            alpha_n = spray.alpha.values[n]
            beta_n  = spray.beta.values[n]
            vol_n   = spray.vol[n]    
            
            num_alpha += alpha_n*vol_n
            num_beta  += beta_n*vol_n
            
        alpha_mean_vw_val.append(num_alpha/sum(spray.vol))
        beta_mean_vw_val.append(num_beta/sum(spray.vol))
        
        
        '''
        alpha_mean_vw_val.append(case[j].alpha_volume_weighted.mean)
        alpha_rms_vw_val.append(case[j].alpha_volume_weighted.std)
        beta_mean_vw_val.append(case[j].beta_volume_weighted.mean)
        beta_rms_vw_val.append(case[j].beta_volume_weighted.std)
        '''
    alpha_mean_cases.append(alpha_mean_val)
    alpha_rms_cases.append(alpha_rms_val)
    beta_mean_cases.append(beta_mean_val)
    beta_rms_cases.append(beta_rms_val)
    
    alpha_mean_vw_cases.append(alpha_mean_vw_val)
    alpha_rms_vw_cases.append(alpha_rms_vw_val)
    beta_mean_vw_cases.append(beta_mean_vw_val)
    beta_rms_vw_cases.append(beta_rms_vw_val)
        


# choose to plot if arithmetic mean or VW
alpha_mean_to_plot = alpha_mean_vw_cases
beta_mean_to_plot  = beta_mean_vw_cases

#%% UG75 plots


 
# alpha mean
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, alpha_mean_to_plot[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, alpha_mean_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylim((1,4.2))
ax.set_ylabel(y_label_alpha_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_alpha_mean.pdf')
plt.show()
plt.close()

# alpha rms
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, alpha_rms_cases[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, alpha_rms_cases[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, alpha_rms_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_alpha_rms.pdf')
#plt.show()
plt.close()



# beta mean
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, beta_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, beta_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, beta_mean_to_plot[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, beta_mean_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylim((0.28,1))
ax.set_ylabel(y_label_beta_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_beta_mean.pdf')
plt.show()
plt.close()

# beta rms
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, beta_rms_cases[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, beta_rms_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, beta_rms_cases[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, beta_rms_vw_cases[i], '--ob')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_beta_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_beta_rms.pdf')
#plt.show()
plt.close()


#%% UG100 plots



# alpha mean
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, alpha_mean_to_plot[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, alpha_mean_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-or', label=labels_OP[i])
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_alpha_mean.pdf')
plt.show()
plt.close()

# alpha rms
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, alpha_rms_cases[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_rms_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, alpha_rms_cases[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, alpha_rms_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, alpha_rms_cases[i], '-or', label=labels_OP[i])
#ax.plot(x_dx10, alpha_rms_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_alpha_rms.pdf')
#plt.show()
plt.close()


# beta mean
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, beta_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, beta_mean_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, beta_mean_to_plot[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, beta_mean_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, beta_mean_to_plot[i], '-or', label=labels_OP[i])
#ax.plot(x_dx10, beta_mean_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_beta_mean)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_beta_mean.pdf')
plt.show()
plt.close()

# beta rms
plt.figure(figsize=figsize_)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, beta_rms_cases[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, beta_rms_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, beta_rms_cases[i], '-ob', label=labels_OP[i])
#ax.plot(x_dx20, beta_rms_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, beta_rms_cases[i], '-or', label=labels_OP[i])
#ax.plot(x_dx10, beta_rms_vw_cases[i], '--or')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_beta_rms)
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_beta_rms.pdf')
#plt.show()
plt.close()

#%% alpha, beta means together


# UG75
plt.figure(figsize=figsize_double)
plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
ax.plot(x_dx10, beta_mean_to_plot[i], '-ok')
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
i = 1
ax  = plt.gca()
ax.plot(x_dx20, alpha_mean_to_plot[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, beta_mean_to_plot[i], '-ob')
#ax.plot(x_dx20, alpha_mean_vw_cases[i], '--ob')
ax.plot([0,100],[1]*2,'--k')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_beta_mean)
#ax.set_yscale('log')
ax.set_ylim(0.2,4.1)
ax.set_yticks([0.25,0.5,0.75, 1,2, 3,4])
ax.grid()
ax.legend(loc='best',ncol=1)
ax.yaxis.set_label_coords(-.1, .35)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug75_both_alpha_beta_mean.pdf')
plt.show()
plt.close()


# UG100 
plt.figure(figsize=figsize_double)
plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 2
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-ok', label=labels_OP[i])
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
ax.plot(x_dx10, beta_mean_to_plot[i], '-ok')
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--ok')
i = 3
ax  = plt.gca()
ax.plot(x_dx20, alpha_mean_to_plot[i], '-ob', label=labels_OP[i])
ax.plot(x_dx20, beta_mean_to_plot[i], '-ob')
#ax.plot(x_dx20, alpha_mean_vw_cases[i], '--ob')
i = 4
ax  = plt.gca()
ax.plot(x_dx10, alpha_mean_to_plot[i], '-or', label=labels_OP[i])
ax.plot(x_dx10, beta_mean_to_plot[i], '-or')
#ax.plot(x_dx10, alpha_mean_vw_cases[i], '--or')
ax.plot([0,100],[1]*2,'--k')
ax.set_xlim(4.5,15.5)
ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_beta_mean)
#ax.set_yscale('log')
ax.set_ylim(0.2,4.1)
ax.set_yticks([0.25,0.5,0.75, 1,2, 3,4])
ax.grid()
ax.legend(loc='best',ncol=1)
ax.yaxis.set_label_coords(-.1, .35)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'ug100_both_alpha_beta_mean.pdf')
plt.show()
plt.close()
