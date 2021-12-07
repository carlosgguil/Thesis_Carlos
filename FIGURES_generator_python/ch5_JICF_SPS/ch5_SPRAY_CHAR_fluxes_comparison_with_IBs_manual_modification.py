"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays



FFIG = 0.5
SCALE_FACTOR = 1e9
PLOT_ADAPTATION_ITERS = True
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 70*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*20,FFIG*16)
figsize_bar = (FFIG*50,FFIG*20)


folder_ibs = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/IBs/'
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/establishment_and_fluxes/'

#%% Load sprays sprays

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

# Recover individual sprays
sps_UG75_DX10_x05 = sprays_list_UG75_DX10[0]
sps_UG75_DX10_x10 = sprays_list_UG75_DX10[1]

sps_UG75_DX20_x05 = sprays_list_UG75_DX20[0]
sps_UG75_DX20_x10 = sprays_list_UG75_DX20[1]
sps_UG75_DX20_x15 = sprays_list_UG75_DX20[2]

sps_UG100_DX10_x05 = sprays_list_UG100_DX10[0]
sps_UG100_DX10_x10 = sprays_list_UG100_DX10[1]

sps_UG100_DX20_x05 = sprays_list_UG100_DX20[0]
sps_UG100_DX20_x10 = sprays_list_UG100_DX20[1]
sps_UG100_DX20_x15 = sprays_list_UG100_DX20[2]

sps_UG100_DX20_NT_x05 = sprays_list_UG100_DX20_NT[0]
sps_UG100_DX20_NT_x10 = sprays_list_UG100_DX20_NT[1]

sprays_x05 = [sps_UG75_DX10_x05,  sps_UG75_DX20_x05, 
              sps_UG100_DX10_x05, sps_UG100_DX20_x05, sps_UG100_DX20_NT_x05]

sprays_x10 = [sps_UG75_DX10_x10,  sps_UG75_DX20_x10, 
              sps_UG100_DX10_x10, sps_UG100_DX20_x10, sps_UG100_DX20_NT_x10]

sprays_x15 = [sps_UG75_DX20_x15,  sps_UG100_DX20_x15]

#%% Parameters

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
labels_= [label_UG75_DX10 , label_UG75_DX20,
                label_UG100_DX10, label_UG100_DX20,
                label_UG100_DX20_NT]
cases_IBS = [label_UG75_DX10 , label_UG75_DX20,
             label_UG100_DX10, label_UG100_DX20]

# Characteristic times to non-dimensionalize
tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584
tau_ph_UG100_DX20_NO_TURB = 0.2584

tau_values_x05_and_x10 = [tau_ph_UG75_DX10 , tau_ph_UG75_DX20,
                          tau_ph_UG100_DX10, tau_ph_UG100_DX20, tau_ph_UG100_DX20_NO_TURB]


tau_values_x15 = [tau_ph_UG75_DX20, tau_ph_UG100_DX20]

# Injected flow rates
Q_inj_UG100 = params_simulation_UG100['Q_inj']*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR
Q_inj_UG75 = params_simulation_UG75['Q_inj']*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR

# IBs
label_Ql_injected = r'$Q_l ~\mathrm{injected}$'
label_x_equal_5  = r'$x = 5~\mathrm{mm}$'
label_x_equal_10 = r'$x = 10~\mathrm{mm}$'
label_x_equal_15 = r'$x = 15~\mathrm{mm}$'


# For bar graphs
barWidth = 0.25
r1 = np.arange(len(cases_IBS))
r2 = np.array([1,3])
pattern_bars_SLI = '-'


#%% Get dimensionless time, SMD and fluxes evolution

tp_x05 = []; SMD_x05 = []; Ql_x05 = [] ; Ql_SLI_x05 = []
tp_x10 = []; SMD_x10 = []; Ql_x10 = [] ; Ql_SLI_x10 = []
tp_x15 = []; SMD_x15 = []; Ql_x15 = [] ; Ql_SLI_x15 = []


# x05 and x10
for i in range(len(sprays_x05)):
    time_x05 = sprays_x05[i].time_instants*1e3/tau_values_x05_and_x10[i]
    time_x05 -= time_x05[0]
    time_x05 += 2
    tp_x05.append(time_x05)
    SMD_x05.append(sprays_x05[i].SMD_evol)
    Ql_x05.append(sprays_x05[i].Q_evol*SCALE_FACTOR)
    Ql_SLI_x05.append(Ql_x05[i][-1])
    
    time_x10 = sprays_x10[i].time_instants*1e3/tau_values_x05_and_x10[i]
    time_x10 -= time_x10[0]
    time_x10 += 2
    tp_x10.append(time_x10)
    SMD_x10.append(sprays_x10[i].SMD_evol)
    Ql_x10.append(sprays_x10[i].Q_evol*SCALE_FACTOR)
    Ql_SLI_x10.append(Ql_x10[i][-1])

# Remove last Ql_SLI for UG100_DX20_NT
Ql_SLI_x05 = Ql_SLI_x05[:-1]
Ql_SLI_x10 = Ql_SLI_x10[:-1]

# x15
for i in range(len(sprays_x15)):
    time_x15 = sprays_x15[i].time_instants*1e3/tau_values_x15[i]
    time_x15 -= time_x15[0]
    time_x15 += 2
    tp_x15.append(time_x15)
    SMD_x15.append(sprays_x15[i].SMD_evol)
    Ql_x15.append(sprays_x15[i].Q_evol*SCALE_FACTOR)
    Ql_SLI_x15.append(Ql_x15[i][-1])




        

#%% IBs

# read dataframes
df_UG100_DX20_x05 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG100_dx20_Q_x05')
df_UG100_DX20_x10 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG100_dx20_Q_x10')
df_UG100_DX20_x15 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG100_dx20_Q_x15')

df_UG100_DX10_x05 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG100_dx10_Q_x05')
df_UG100_DX10_x10 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG100_dx10_Q_x10')

df_UG75_DX20_x05 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG75_dx20_Q_x05')
df_UG75_DX20_x10 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG75_dx20_Q_x10')
df_UG75_DX20_x15 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG75_dx20_Q_x15')

df_UG75_DX10_x05 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG75_dx10_Q_x05')
df_UG75_DX10_x10 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/uG75_dx10_Q_x10')

# get mean fluxes
Q_mean_UG75_DX10_x05 = df_UG75_DX10_x05['Q_t_x05_mean_evol'].values
Q_mean_UG75_DX10_x10 = df_UG75_DX10_x10['Q_t_x10_mean_evol'].values
Q_mean_UG75_DX20_x05 = df_UG75_DX20_x05['Q_t_x05_mean_evol'].values
Q_mean_UG75_DX20_x10 = df_UG75_DX20_x10['Q_t_x10_mean_evol'].values
Q_mean_UG75_DX20_x15 = df_UG75_DX20_x15['Q_t_x15_mean_evol'].values
Q_mean_UG100_DX10_x05 = df_UG100_DX10_x05['Q_t_x05_mean_evol'].values
Q_mean_UG100_DX10_x10 = df_UG100_DX10_x10['Q_t_x10_mean_evol'].values
Q_mean_UG100_DX20_x05 = df_UG100_DX20_x05['Q_t_x05_mean_evol'].values
Q_mean_UG100_DX20_x10 = df_UG100_DX20_x10['Q_t_x10_mean_evol'].values
Q_mean_UG100_DX20_x15 = df_UG100_DX20_x15['Q_t_x15_mean_evol'].values

Q_x_mean_x05 = [Q_mean_UG75_DX10_x05[-1],Q_mean_UG75_DX20_x05[-1],
                     Q_mean_UG100_DX10_x05[-1],Q_mean_UG100_DX20_x05[-1]]
Q_x_mean_x10 = [Q_mean_UG75_DX10_x10[-1],Q_mean_UG75_DX20_x10[-1],
                     Q_mean_UG100_DX10_x10[-1],Q_mean_UG100_DX20_x10[-1]]
Q_x_mean_x15 = [Q_mean_UG75_DX20_x15[-1],
                     Q_mean_UG100_DX20_x15[-1]]


#%% Bar graph

# modify sli values

# x = 05 mm
Ql_SLI_x05[0] = 2700.23
Ql_SLI_x05[1] = 2580.0
Ql_SLI_x05[2] = 3591.3
Ql_SLI_x05[3] = 3600


# x = 10 mm
Ql_SLI_x10[2] = 3430.0
Ql_SLI_x10[3] = 3200

'''
# x = 15 mm
Ql_SLI_x15[0] = 1650
Ql_SLI_x15[1] = 2450
'''

# plot
plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.plot([r1[0]-barWidth*1.5,r2[0]+barWidth*1.5],[Q_inj_UG75]*2, '--k', label=label_Ql_injected,linewidth=4*FFIG)
plt.plot([r1[2]-barWidth*1.5,r2[-1]+barWidth*1.5],[Q_inj_UG100]*2, '--k',linewidth=4*FFIG)
# x = 05
plt.bar(r1-barWidth*1.1, Q_x_mean_x05, width=barWidth/2, color='blue', edgecolor='white', label=label_x_equal_5)
plt.bar(r1-barWidth*1.2/2, Ql_SLI_x05, width=barWidth/2, color='blue', edgecolor='black',  hatch = pattern_bars_SLI)
# x = 10
plt.bar(r1, Q_x_mean_x10, width=barWidth/2, color='grey', edgecolor='white', label=label_x_equal_10)
plt.bar(r1+barWidth/2, Ql_SLI_x10, width=barWidth/2, color='grey', edgecolor='black', hatch = pattern_bars_SLI)
# x = 15
plt.bar(r2+barWidth*1.1, Q_x_mean_x15, width=barWidth/2, color='red', edgecolor='white', label=label_x_equal_15)
plt.bar(r2+barWidth*1.6, Ql_SLI_x15, width=barWidth/2, color='red', edgecolor='black', hatch = pattern_bars_SLI)
#plt.xlabel('Case')#, fontweight='bold')
plt.ylabel(y_label_Ql)
plt.xticks([r for r in range(len(cases_IBS))], cases_IBS)
plt.legend(loc='upper left', ncol=2)
plt.tight_layout()
#plt.savefig(folder_manuscript+'fluxes_SLI_vs_IBs.pdf')
plt.show()
plt.close()    
            