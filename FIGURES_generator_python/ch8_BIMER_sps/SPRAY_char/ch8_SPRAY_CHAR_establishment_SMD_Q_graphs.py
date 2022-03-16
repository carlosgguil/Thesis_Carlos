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
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*25,FFIG*22)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/establishment_and_fluxes/'

#%% Load sprays

# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    

# Load sprays
sp1, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation)

sprays_list_all = [sp3, sp2, sp1]

#%% Parameters


format_separating_line = 'k'
linewidth_separating_line = 15*FFIG
linewidth_Ql = 6*FFIG

# axis labels
x_label_time  = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
y_label_SMD   = r'$\mathrm{SMD}~[\mu \mathrm{m}]$'
y_label_Ql    = r'$Q_l~[\mathrm{mm}^3~\mathrm{s}^{-1}]$'

# legend labels
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07 = r'$\mathrm{DX}07$'
labels_title = [label_DX15 , label_DX10,label_DX07]


label_xD03p33 = r'$x_c/d_\mathrm{inj} = 3.33$'
label_xD05p00 = r'$x_c/d_\mathrm{inj} = 5.00$'
label_xD06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'
labels_ = [label_xD03p33, label_xD05p00, label_xD06p67]

# Characteristic times to non-dimensionalize
tau_dr_DX15 = 0.562
tau_dr_DX10 = 0.354
tau_dr_DX07 = 0.359

tau_values = [tau_dr_DX15 , tau_dr_DX10, tau_dr_DX07]

# Injected flow rates
d_inj = 0.3E-3
Q_inj= np.pi/4*d_inj**2*2.6*SCALE_FACTOR 

# shifting times
tp_0_true_values = False


if tp_0_true_values:
    tp_0_DX07 = 0.9310/tau_dr_DX07
else:
    tp_0_DX07 = 2.05

# these are ~ 2 anyways
tp_0_DX10 = 0.7688/tau_dr_DX10
tp_0_DX15 = 1.1693/tau_dr_DX15 

    
# define maximum values for t' (obtained from ch8_nelem_plot.py)
tp_max_DX15 = 6.775423875670118 # diff of 1*tp
tp_max_DX10 = 4.88789371578306 
tp_max_DX07 = 3.9651507666425956 


tp_0_cases = [tp_0_DX15, tp_0_DX10, tp_0_DX07]
tp_max_cases =  [tp_max_DX15, tp_max_DX10, tp_max_DX07]


    

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
        
        #t_plot_i = time
        time_val.append(t_plot_i)
        SMD_val.append(case[j].SMD_evol)
        Ql_val.append(case[j].Q_evol*SCALE_FACTOR)
    tp_cases.append(time_val)
    SMD_cases.append(SMD_val)
    Ql_cases.append(Ql_val)
        


#%% Plots (Ql above, SMD below)

# DX15
i = 0
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj]*2, '--k',linewidth = linewidth_Ql)
# xD = 5
j = 1 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# xD = 6.67
j = 2
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,3,4,5,6,7]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-400,400)
ax.set_yticks([0, 100, 200, 300, 400])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(40,100)
ax2.set_yticks([40,50,60,70])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_DX15.pdf')
plt.show()
plt.close()







#%% DX10


i = 1
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj]*2, '--k',linewidth = linewidth_Ql)
# xD = 5
j = 1 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# xD = 6.67
j = 2 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'b', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'b', label=labels_[j]) 
# Raya horizontal y parametros a tunear
ax.plot([0,100],[0]*2,format_separating_line,linewidth=linewidth_separating_line)
ax.set_xlabel(x_label_time)
x_lim_ = tp_cases[i][0][0]-0.05,tp_cases[i][0][-1]+0.05
x_ticks_ = [2,3,4,5]
ax.set_xlim(x_lim_)
ax2.set_xlim(x_lim_)
ax.set_xticks(x_ticks_)
ax2.set_xticks(x_ticks_)

ax.set_ylabel(y_label_Ql)
ax.set_ylim(-400,400)
ax.set_yticks([0, 100, 200, 300, 400])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(30,70)
ax2.set_yticks([30,35,40,45,50])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
# Plot just to add legend
ax2.plot((0,0),(0,1), '--k',linewidth = linewidth_Ql, label=r'$Q_l~\mathrm{injected}$')
ax2.legend(loc='upper right',ncol=2)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_DX10.pdf')
plt.show()
plt.close()





#%% DX07


i = 2
plt.figure(figsize=figsize_)
plt.title(labels_title[i])
ax  = plt.gca()
ax2 = ax.twinx()
# Flujo inyectado
ax.plot([tp_cases[i][0][0],tp_cases[i][0][-2]], [Q_inj]*2, '--k',linewidth = linewidth_Ql)
# xD = 5
j = 1 
ax.plot(tp_cases[i][j], Ql_cases[i][j], 'k', label=labels_[j]) 
ax2.plot(tp_cases[i][j], SMD_cases[i][j], 'k', label=labels_[j])
# xD = 6.67
j = 2
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
ax.set_ylim(-300,300)
ax.set_yticks([0, 100, 200, 300])
ax.yaxis.set_label_coords(-0.15,0.8)

ax2.set_ylabel(y_label_SMD)
#ax2.set_ylim(0,300)
#ax2.set_yticks([0,50,100,150])
ax2.set_ylim(30,70)
ax2.set_yticks([30,35,40,45,50])
ax2.yaxis.set_label_coords(1.1,0.224)

ax.grid()
ax2.grid()
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'establishment_DX07.pdf')
plt.show()
plt.close()




#%% Plot acumulation times, N_dr and N_dr per tp
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


