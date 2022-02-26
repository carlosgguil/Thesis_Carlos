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
figsize_ = (FFIG*25,FFIG*22)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/establishment_and_fluxes/'

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
y_label_SMD   = r'$\mathrm{SMD}~[\mu \mathrm{m}]$'
y_label_Ql    = r'$Q_l~[\mathrm{mm}^3~\mathrm{s}^{-1}]$'

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

# Injected flow rates
d_inj = 0.45E-3
Q_inj_UG100 = np.pi/4*d_inj**2*23.33*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR
Q_inj_UG75 = np.pi/4*d_inj**2*17.5*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR

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
    

#%% Get dimensionless time, SMD and fluxes evolution


tp_cases = []; SMD_cases = []; Ql_cases = []
for i in range(len(sprays_list_all)):
    case = sprays_list_all[i]
    tau_char = tau_values[i]
    time_val = []; SMD_val = []; Ql_val = []
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
        SMD_val.append(case[j].SMD_evol)
        Ql_val.append(case[j].Q_evol*SCALE_FACTOR)
    tp_cases.append(time_val)
    SMD_cases.append(SMD_val)
    Ql_cases.append(Ql_val)
        


#%% Plots (Ql above, SMD below)

# UG75_DX10
i = 0
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj_UG75]*2, '--k',linewidth = linewidth_Ql)
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,3,4]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-5500,5250)
ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(30,270)
ax2.set_yticks([50,75,100,125,150])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX10.pdf')
plt.show()
plt.close()







#%%

# UG75_DX20
i = 1
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj_UG75]*2, '--k',linewidth = linewidth_Ql, label=r'$Q_l~\mathrm{injected}$')
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# x = 15 mm
j = 2
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'r', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'r', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-5500,6500)
ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000,6000])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(50,540)
ax2.set_yticks(np.linspace(50,250,6))
ax2.yaxis.set_label_coords(1.11,0.224)

ax.grid()
ax2.grid()
#ax.legend(loc='upper right',ncol=2)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX20.pdf')
plt.show()
plt.close()




#%%

# UG100_DX10
i = 2
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj_UG100]*2, '--k',linewidth = linewidth_Ql)
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,2.5,3,3.5]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-5000,5000)
ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(50,225)
ax2.set_yticks([50,75,100,125])
ax2.yaxis.set_label_coords(1.11,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX10.pdf')
plt.show()
plt.close()



#%%




# UG100_DX20
i = 3
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj_UG100]*2, '--k',linewidth = linewidth_Ql)
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# x = 15 mm
j = 2
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'r', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'r', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-6000,6000)
ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6e3])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
ax2.set_ylim(50,350)
ax2.set_yticks(np.linspace(50,200,6))
ax2.yaxis.set_label_coords(1.11,0.224)

ax.grid()
ax2.grid()
# Plot just to add legend
ax2.plot((0,0),(0,1), '--k',linewidth = linewidth_Ql, label=r'$Q_l~\mathrm{injected}$')
ax2.legend(loc='lower right',ncol=2)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20.pdf')
plt.show()
plt.close()


#%%


# UG100_DX20_NT
i = 4
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj_UG100]*2, '--k',linewidth = linewidth_Ql)
# x = 05 mm
j = 0 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# x = 10 mm
j = 1; tp0_NT = 6
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j][tp0_NT:], SMD_cases[i][j][tp0_NT:], 'b', label=labels_[j]) 

ax.grid(zorder=1)
ax2.grid(zorder=2)
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line,zorder=3)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
#x_ticks_ = [2,2.5,3]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
#ax.set_xticks(x_ticks_)
#ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-10000,10000)
ax.set_yticks([0, 2000, 4000, 6000, 8000, 10e3])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
ax2.set_ylim(60,417)
ax2.set_yticks(np.linspace(60,240,7))
ax2.yaxis.set_label_coords(1.11,0.224)

plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG100_DX20_NT.pdf')
plt.show()
plt.close()

# Plot acumulation times, N_dr and N_dr per tp
for i in range(len(sprays_list_all)):
    print('\nCase '+labels_title[i])
    tp_acc_ = tp_cases[i][0][-1] - tp_cases[i][0][0]
    t_acc_ = tp_acc_*tau_values[i]
    print(f'   t_acc = {t_acc_}')
    print(f'   tp_acc = {tp_acc_}')
    sprays = sprays_list_all[i]
    for j in range(len(sprays)):
        spray = sprays[j]
        print(f' Plane {j}: N_dr = {spray.n_droplets} ; N_dr/tp = {spray.n_droplets/tp_acc_:.3f} ; N_dr/t = {spray.n_droplets/t_acc_:.3f}')


#%% Plots (SMD above, Ql below)
'''
# UG75_DX10
i = 0
plt.figure(figsize=figsize_)
# x = 05 mm
j = 0
plt.plot(tp_cases[i][j], SMD_cases[i][j], color='black',label=labels_[j]) 
# x = 10 mm
j = 1
plt.plot(tp_cases[i][j], SMD_cases[i][j], color='blue',label=labels_[j]) 
plt.xlabel(x_label_time)
plt.ylabel(y_label_SMD)
#plt.xticks([0,5,10,15,20])
plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_xb+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
#plt.savefig(folder_manuscript+'SMD_evol_at_x05.pdf')
plt.show()
plt.close()
'''

'''
# UG75_DX10
i = 0
plt.figure(figsize=figsize_)
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0
ax.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,color='grey')
ax.set_xlabel(x_label_time)
ax.set_xlim(tp_cases[i][j][0]-0.05,tp_cases[i][j][-1]+0.05)
ax.set_xticks([2,3,4])
ax.set_ylabel(y_label_SMD)
ax.set_ylim(-150,150)
ax.set_yticks([0,50,100,150])
ax.yaxis.set_label_coords(-0.15,0.8)
ax2.set_ylabel(y_label_Ql)
ax2.set_ylim(0,10000)
ax2.set_yticks([0,1000, 2000, 3000, 4000])
ax2.yaxis.set_label_coords(1.15,0.224)
ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX10.pdf')
plt.show()
plt.close()


# UG75_DX20
i = 1
plt.figure(figsize=figsize_)
ax  = plt.gca()
ax2 = ax.twinx()
# x = 05 mm
j = 0
ax.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
# x = 10 mm
j = 1
ax.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
# x = 15 mm
j = 2
ax.plot(tp_cases[i][j], SMD_cases[i][j], 'r', label=labels_[j]) 
ax2.plot(tp_cases[i][j], Ql_cases[i][j], 'r', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,color='grey')
ax.set_xlabel(x_label_time)
ax.set_xlim(tp_cases[i][j][0]-0.05,tp_cases[i][j][-1]+0.05)
ax.set_xticks([2,3,4])
ax.set_ylabel(y_label_SMD)
ax.set_ylim(-200,200)
ax.set_yticks([0,50,100,150])
ax.yaxis.set_label_coords(-0.15,0.8)
ax2.set_ylabel(y_label_Ql)
ax2.set_ylim(0,10000)
ax2.set_yticks([0,1000, 2000, 3000, 4000])
ax2.yaxis.set_label_coords(1.15,0.224)
ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_UG75_DX20.pdf')
plt.show()
plt.close()
'''



