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
figsize_double = (FFIG*23.5,FFIG*18.5)


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/deformation/'

#%% Load sprays

# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    
sampling_planes = ['xD_05p00','xD_06p67',
                   'xD_08p33','xD_10p00']

# Load sprays
_, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation, sampling_planes = sampling_planes)

sprays_list_all = [sp2, sp3]


#%% Parameters


format_separating_line = 'k'
linewidth_separating_line = 15*FFIG
linewidth_Ql = 6*FFIG

# axis labels
#x_label_  = r'$x_c/d_\mathrm{inj}$' #r'$t~[\mathrm{ms}]$'
x_label_ = r'$x_c ~\left[ \mathrm{mm} \right]$'
y_label_alpha_mean = r'$\overline{\alpha}_\mathrm{VW}$'
y_label_alpha_rms  = r'$\alpha_\mathrm{RMS}$'
y_label_beta_mean = r'$\overline{\beta}_\mathrm{VW}$'
y_label_beta_rms = r'$\beta_\mathrm{RMS}$'
y_label_alpha_beta_mean = r'$\overline{\alpha}_\mathrm{VW}, \overline{\beta}_\mathrm{VW}$'


# legend labels
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07 = r'$\mathrm{DX}07$'
labels_OP = [label_DX10 , label_DX15]



label_xD03p33 = r'$x_c/d_\mathrm{inj} = 3.33$'
label_xD05p00 = r'$x_c/d_\mathrm{inj} = 5.00$'
label_xD06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'
labels_ = [label_xD03p33, label_xD05p00, label_xD06p67]
# x coordinates
xD = np.array([5,6.67,8.33,10])*0.3


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

#%% alpha, beta means together



# 
plt.figure(figsize=figsize_double)
#plt.title(r'$u_g = 75~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
i = 0
ax  = plt.gca()
ax.plot(xD, alpha_mean_to_plot[i], '-ob', label=labels_OP[i])
ax.plot(xD, beta_mean_to_plot[i], '-ob')
i = 1
ax  = plt.gca()
ax.plot(xD, alpha_mean_to_plot[i], '-ok', label=labels_OP[i])
ax.plot(xD, beta_mean_to_plot[i], '-ok')
ax.plot([0,100],[1]*2,'--k')
ax.set_xlim(1.4,3.1)
#ax.set_xticks([5,10,15])
ax.set_xlabel(x_label_)
ax.set_ylabel(y_label_alpha_beta_mean)
#ax.set_yscale('log')
ax.set_ylim(0.2,2.5)
#ax.set_yticks([0.25,0.5,0.75, 1,2, 3,4])
ax.grid()
ax.legend(loc='best',ncol=1)
plt.tight_layout(pad=0)
plt.savefig(folder_manuscript+'deformation_both_alpha_beta_mean.pdf')
plt.show()
plt.close()


