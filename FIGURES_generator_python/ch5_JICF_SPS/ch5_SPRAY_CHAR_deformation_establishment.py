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
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*25,FFIG*18)
#figsize_ = (FFIG*26,FFIG*16)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/deformation_establishment/'

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
x_label_time  = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
y_label_alpha_mean = r'$\overline{\alpha}_\mathrm{VW}$'
y_label_alpha_rms  = r'$\alpha_\mathrm{RMS,VW}$'
y_label_beta_mean = r'$\overline{\beta}_\mathrm{VW}$'
y_label_beta_rms = r'$\beta_\mathrm{RMS,VW}$'


# legend labels
label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_UG100_DX20_NT = r'$\mathrm{UG100}\_\mathrm{DX20}\_\mathrm{NT}$'
labels_title = [label_UG75_DX10 , label_UG75_DX20,
                label_UG100_DX10, label_UG100_DX20,
                label_UG100_DX20_NT]
cases_IBS = [label_UG75_DX10 , label_UG75_DX20,
             label_UG100_DX10, label_UG100_DX20]


label_x05 = r'$x = 5~\mathrm{mm}$'
label_x10 = r'$x = 10~\mathrm{mm}$'
label_x15 = r'$x = 15~\mathrm{mm}$'
labels_ = [label_x05, label_x10, label_x15]

# Characteristic times to non-dimensionalize
tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584
tau_ph_UG100_DX20_NO_TURB = 0.2602

tau_values = [tau_ph_UG75_DX10 , tau_ph_UG75_DX20,
              tau_ph_UG100_DX10, tau_ph_UG100_DX20, tau_ph_UG100_DX20_NO_TURB]

# shifting times
tp_0_true_values = False

if tp_0_true_values:
    tp_0_UG100_DX20 = 0.6840/tau_ph_UG100_DX20
    tp_0_UG100_DX20_NT = 0.7844/tau_ph_UG100_DX20_NO_TURB
    tp_0_UG100_DX10 = 0.4173/tau_ph_UG100_DX10
    tp_0_UG75_DX20 = 0.9640/tau_ph_UG75_DX20
    tp_0_UG75_DX10 = 0.5032/tau_ph_UG75_DX10
else:
    tp_0_UG100_DX20 = 2.3
    tp_0_UG100_DX20_NT = 2.2
    tp_0_UG100_DX10 = 1.9080932784636488
    tp_0_UG75_DX20 = 2.3
    tp_0_UG75_DX10 = 1.85
    
# define maximum values for t' (obtained from ch5_nelem_plot.py)
tp_max_UG100_DX20 = 23.8370270548746 # diff of 1*tp
tp_max_UG100_DX20_NT = 23.436246263752494 # diff of 2*tp
tp_max_UG100_DX10 = 3.5785496471047074 # diff of 0.5*tp
tp_max_UG75_DX20 = 17.695510529612424 # no diff !
tp_max_UG75_DX10 = 3.6428757530101596 # no diff !

tp_0_cases = [tp_0_UG75_DX10, tp_0_UG75_DX20,
              tp_0_UG100_DX10, tp_0_UG100_DX20, tp_0_UG100_DX20_NT]

tp_max_cases =  [tp_max_UG75_DX10, tp_max_UG75_DX20,
                 tp_max_UG100_DX10, tp_max_UG100_DX20, tp_max_UG100_DX20_NT]


#%% Get dimensionless time and velocities evolution


tp_cases = []; 
alpha_mean_cases = []; uy_mean_cases = []; beta_mean_cases = []
alpha_rms_cases  = []; uy_rms_cases  = []; beta_rms_cases  = []
for i in range(len(sprays_list_all)):
    case = sprays_list_all[i]
    tau_char = tau_values[i]
    time_val = []; 
    alpha_mean_val = []; uy_mean_val = []; beta_mean_val = []
    alpha_rms_val = []; uy_rms_val = []; beta_rms_val = []
    # to shift time
    tp_0_i = tp_0_cases[i]
    tp_max_i = tp_max_cases[i]
    for j in range(len(case)):
        time = case[j].time_instants*1e3/tau_char
        time -= time[0]
        time += 2
        
        # Shift time
        m_ij = (tp_max_i - tp_0_i)/(time[-1] - time[0])
        t_plot_i = m_ij*(time - time[0]) + tp_0_i
        time_val.append(t_plot_i)
        alpha_mean_val.append(case[j].alpha_mean_vw_evol)
        alpha_rms_val.append(case[j].alpha_rms_vw_evol)
        uy_mean_val.append(case[j].uy_mean_evol)
        uy_rms_val.append(case[j].uy_rms_evol)
        beta_mean_val.append(case[j].beta_mean_vw_evol)
        beta_rms_val.append(case[j].beta_rms_vw_evol)
    tp_cases.append(time_val)
    alpha_mean_cases.append(alpha_mean_val)
    alpha_rms_cases.append(alpha_rms_val)
    uy_mean_cases.append(uy_mean_val)
    uy_rms_cases.append(uy_rms_val)
    beta_mean_cases.append(beta_mean_val)
    beta_rms_cases.append(beta_rms_val)
        




# Plots 
#%% UG75_DX10 
 
# UG75_DX10 mean
i = 0
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], alpha_mean_cases[i][j], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_mean_cases[i][j], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j], beta_mean_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], alpha_mean_cases[i][j], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_mean_cases[i][j], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j], beta_mean_cases[i][j], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,3,4]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)


ax.set_ylabel(y_label_alpha_mean)
ax.set_ylim(0,5)
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.set_yticks([1, 2, 3, 4,5])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_mean)
ax2.set_ylim(0,5)
ax2.set_yticks([0, 0.5, 1])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX10_mean.pdf')
plt.show()
plt.close()


# UG75_DX10 rms
i = 0
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], alpha_rms_cases[i][j], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_rms_cases[i][j], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j], beta_rms_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], alpha_rms_cases[i][j], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_rms_cases[i][j], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j], beta_rms_cases[i][j], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,3,4]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_rms)
ax.set_ylim(-4,4)
ax.set_yticks([0, 1, 2, 3, 4])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_rms)
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.25, 0.5])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX10_rms.pdf')
plt.show()
plt.close()




#%% UG75_DX20 

# UG75_DX20 mean
i = 1; tp0 = 10
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'b', label=labels_[j])
# x = 15 mm
j = 2; 
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'r', label=labels_[j]) 
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'r', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_mean)
ax.set_ylim(0,3.5)
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_mean)
ax2.set_ylim(0,3.5)
ax2.set_yticks([0, 0.5, 1])

#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
ax.legend(loc='center right')
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX20_mean.pdf')
#plt.show()
plt.close()








# UG75_DX20 rms
i = 1; tp0 = 15
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'b', label=labels_[j])
# x = 15 mm
j = 2; 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'r', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--r', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'r', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_rms)
ax.set_ylim(-2,2)
ax.set_yticks([0, 1, 2])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_rms)
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.25, 0.5])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
ax.legend(loc='center right')
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX20_rms.pdf')
plt.show()
plt.close()


#%% UG100_DX10 

# UG100_DX10 mean
i = 2; tp0 = 1
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], alpha_mean_cases[i][j], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_mean_cases[i][j], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j], beta_mean_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], alpha_mean_cases[i][j], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j], uy_mean_cases[i][j], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_mean)
ax.set_ylim(0,3.5)
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_mean)
ax2.set_ylim(-1,1)
#ax2.set_yticks([-5, 0,5,10])
ax2.set_ylim(0,3.5)
ax2.set_yticks([0, 0.5, 1])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX10_mean.pdf')
plt.show()
plt.close()




# UG100_DX10 rms
i = 2; tp0 = 1
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)


ax.set_ylabel(y_label_alpha_rms)
ax.set_ylim(-2,2)
ax.set_yticks([0, 1, 2])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_rms)
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.25, 0.5])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX10_rms.pdf')
plt.show()
plt.close()












#%% UG100_DX20 


# UG100_DX20 mean
i = 3; tp0 = 5
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_mean_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_mean_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'b', label=labels_[j])
# x = 15 mm
j = 2; 
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'r', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_mean_cases[i][j][tp0:], '--r', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'r', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)


ax.set_ylabel(y_label_alpha_mean)
ax.set_ylim(0,3.5)
ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_mean)
ax2.set_ylim(0,3.5)
ax2.set_yticks([0, 0.5, 1])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20_mean.pdf')
plt.show()
plt.close()





# UG100_DX20 rms
i = 3; tp0 = 5
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'b', label=labels_[j])
# x = 15 mm
j = 2; 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'r', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--r', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'r', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_rms)
ax.set_ylim(-2,2)
ax.set_yticks([0, 1, 2])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_rms)
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.25, 0.5])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20_rms.pdf')
plt.show()
plt.close()




#%% UG100_DX20_NT 




# UG100_DX20_NT mean
i = 4; tp0 = 0
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_mean_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_mean_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_mean_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_mean_cases[i][j][tp0:], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[1]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_mean)
ax.set_ylim(0,4)
ax.set_yticks([1, 2, 3, 4])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_mean)
ax2.set_ylim(0,4)
ax2.set_yticks([0, 0.5, 1])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20_NT_mean.pdf')
#plt.show()
plt.close()



# UG100_DX20_NT rms
i = 4; tp0 = 20
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'k', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--k', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j][tp0:], alpha_rms_cases[i][j][tp0:], 'b', label=labels_[j]) 
#ax2.plot(tp_cases[i][j][tp0:], uy_rms_cases[i][j][tp0:], '--b', label=labels_[j])
ax2.plot(tp_cases[i][j][tp0:], beta_rms_cases[i][j][tp0:], 'b', label=labels_[j])
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_alpha_rms)
ax.set_ylim(-2,2)
ax.set_yticks([0, 1, 2])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_beta_rms)
ax2.set_ylim(0,1)
ax2.set_yticks([0, 0.25, 0.5])
#ax2.set_ylim(30,270)
#ax2.set_yticks([50,100,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20_NT_rms.pdf')
plt.show()
plt.close()






