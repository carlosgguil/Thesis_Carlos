"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('../..')
from sli_functions import load_all_BIMER_global_sprays



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
figsize_bar = (FFIG*40,FFIG*16)


folder_ibs = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/IBs/'
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/establishment_and_fluxes/'

#%% Load sprays sprays


# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']    

# Load sprays
_, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation)

sprays_list_DX15 = sp2
sprays_list_DX10 = sp3


#%%
# Recover individual sprays
sps_DX15_xD05p00 = sprays_list_DX15[1]
sps_DX15_xD06p67 = sprays_list_DX15[2]

sps_DX10_xD05p00 = sprays_list_DX10[1]
sps_DX10_xD06p67 = sprays_list_DX10[2]


sprays_xD05p00 = [sps_DX15_xD05p00,  sps_DX10_xD05p00]
sprays_xD06p67 = [sps_DX15_xD06p67,  sps_DX10_xD06p67]


Ql_SLI_xD05p00 = np.array([sps_DX10_xD05p00.Q, sps_DX15_xD05p00.Q])*SCALE_FACTOR
Ql_SLI_xD06p67 = np.array([sps_DX10_xD06p67.Q, sps_DX15_xD06p67.Q])*SCALE_FACTOR



#%% Parameters

# axis labels
y_label_Ql    = r'$Q_l~[\mathrm{mm}^3~\mathrm{s}^{-1}]$'

# legend labels
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
labels_ = [label_DX10,label_DX15]
cases_IBS = labels_



# Injected flow rates
Q_inj = params_simulation['Q_inj']*SCALE_FACTOR #3.6700294207081691E-006*SCALE_FACTOR

# IBs
label_Ql_injected = r'$Q_l ~\mathrm{injected}$'
label_xD05p00 = r'$x_c/d_\mathrm{inj} = 5.00$'
label_xD06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'

# For bar graphs
barWidth = 0.25
r1 = np.arange(len(labels_))
pattern_bars_SLI = '-'



        

#%% IBs

# read dataframes
df_DX15_xD05p00 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/dx15p0_Q_xD_05p00.csv')
df_DX15_xD06p67 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/dx15p0_Q_xD_06p67.csv')

df_DX10_xD05p00 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/dx10p0_Q_xD_05p00.csv')
df_DX10_xD06p67 = pd.read_csv(folder_ibs+'/overall_integrated_fluxes/dx10p0_Q_xD_06p67.csv')



# get mean fluxes
Q_mean_DX15_xD05p00 = df_DX15_xD05p00['Q_t_xD_05p00_mean_evol'].values
Q_mean_DX15_xD06p67 = df_DX15_xD06p67['Q_t_xD_06p67_mean_evol'].values

Q_mean_DX10_xD05p00 = df_DX10_xD05p00['Q_t_xD_05p00_mean_evol'].values
Q_mean_DX10_xD06p67 = df_DX10_xD06p67['Q_t_xD_06p67_mean_evol'].values




Q_x_mean_xD05p00 = [Q_mean_DX10_xD05p00[-1], Q_mean_DX15_xD05p00[-1]]
Q_x_mean_xD06p67 = [Q_mean_DX10_xD06p67[-1], Q_mean_DX15_xD06p67[-1]]

#%% 

# Bar graph
plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.plot([r1[0]-barWidth*1.1,r1[-1]+barWidth*1.],[Q_inj]*2, '--k', label=label_Ql_injected,linewidth=4*FFIG)
# xD = 05.00
plt.bar(r1-barWidth*0.8, Q_x_mean_xD05p00, width=barWidth/2, color='blue', edgecolor='white', label=label_xD05p00)
plt.bar(r1-barWidth*0.3, Ql_SLI_xD05p00, width=barWidth/2, color='blue', edgecolor='black', hatch = pattern_bars_SLI)
# xD = 06.67
plt.bar(r1+barWidth*0.3, Q_x_mean_xD06p67, width=barWidth/2, color='red', edgecolor='white', label=label_xD06p67)
plt.bar(r1+barWidth*0.8, Ql_SLI_xD06p67, width=barWidth/2, color='red', edgecolor='black', hatch = pattern_bars_SLI)
#plt.xlabel('Case')#, fontweight='bold')
plt.ylabel(y_label_Ql)
plt.xticks([r for r in range(len(cases_IBS))], cases_IBS)
plt.legend(loc='upper left', ncol=3)
plt.tight_layout()
plt.ylim(0,250)
plt.savefig(folder_manuscript+'fluxes_SLI_vs_IBs.pdf')
plt.show()
plt.close()   
            