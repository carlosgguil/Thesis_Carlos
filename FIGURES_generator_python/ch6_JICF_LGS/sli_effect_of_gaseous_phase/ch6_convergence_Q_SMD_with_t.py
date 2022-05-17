
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""





import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions_expe_comparison import average_along_y, average_along_z
from functions_expe_comparison import get_SMD_from_integrated_profile, get_SMD_flux_weighted
from sprPost_calculations import get_discrete_spray, get_sprays_list
from sprPost_functions import get_grids_common_boundaries
import sprPost_plot as sprPlot


# Change size of figures 
FFIG = 0.5
#mpl.rcParams['font.size'] = 40*fPic
plt.rcParams['xtick.labelsize']  = 70*FFIG
plt.rcParams['ytick.labelsize']  = 70*FFIG
plt.rcParams['axes.labelsize']   = 60*FFIG
plt.rcParams['axes.labelpad']    = 30*FFIG
plt.rcParams['axes.titlesize']   = 50*FFIG
plt.rcParams['legend.fontsize']  = 40*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 20*FFIG #20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*18,FFIG*13)
figsize_expe = (FFIG*21.7,FFIG*12.4)
figsize_along_z  = (FFIG*10,FFIG*15)


label_time = r'$t$~[$\mathrm{ms}$]'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'
label_Q = r'$Q_l$ [$\mathrm{mm}^3~\mathrm{s}^{-1}$]'



folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'






cases = ['APTE', 'full_ALM_initial', 'full_ALM_FDC_0p24', 'full_ALM_FDC_0p30']


folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_full_no_ALM  = folder + '/FULL_domain/LGS_no_ALM/store_variables/'
folder_full_ALM_initial = folder + '/FULL_domain/LGS_ALM_initial/store_variables/'
folder_full_FDC_0p10 = folder + '/FULL_domain/LGS_ALM_FDC_0p10/store_variables/'
folder_full_FDC_0p24 = folder + '/FULL_domain/LGS_ALM_FDC_0p24/store_variables/'
folder_full_FDC_0p30 = folder + '/FULL_domain/LGS_ALM_FDC_0p30/store_variables/'

folders = {'APTE': folder_APTE ,
           #'full_no_ALM': folder_full_no_ALM,
           'full_ALM_initial': folder_full_ALM_initial,
           #'full_ALM_FDC_0p10': folder_full_FDC_0p10,
           'full_ALM_FDC_0p24': folder_full_FDC_0p24,
           'full_ALM_FDC_0p30': folder_full_FDC_0p30}



label_APTE = r'$\mathrm{Prescribed}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'
label_full_ALM_FDC_0p10 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p24 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p30 = r'$\mathrm{ALM~modified}$'

labels = {'APTE': label_APTE ,
           #'full_no_ALM': label_full_no_ALM,
           'full_ALM_initial': label_full_ALM_initial,
           #'full_ALM_FDC_0p10': label_full_ALM_FDC_0p10,
           'full_ALM_FDC_0p24': label_full_ALM_FDC_0p24,
           'full_ALM_FDC_0p30': label_full_ALM_FDC_0p30}



formats = {'APTE':'-ok', 
           #'full_no_ALM':'-^b', 
           'full_ALM_initial':'-*r',
           #'full_ALM_FDC_0p10':'-*y',
           'full_ALM_FDC_0p24':'-*y',
           'full_ALM_FDC_0p30':'-*g',}

formats = {'APTE':'-k', 
           #'full_no_ALM':'-b', 
           'full_ALM_initial':'-r',
           #'full_ALM_FDC_0p10':'-y',
           'full_ALM_FDC_0p24':'-y',
           'full_ALM_FDC_0p30':'-g'}

tau_x80 = {'APTE':'-k', 
           #'full_no_ALM':'-b', 
           'full_ALM_initial':'-r',
           #'full_ALM_FDC_0p10':'-y',
           'full_ALM_FDC_0p24':'-y',
           'full_ALM_FDC_0p30':'-g'}

#%% Experimental data and simulation parameters (do not touch)



params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']
Q_inj = params_simulation['Q_inj'] *1e9







#%% get data

file = 'convergence_x80mm_Q_SMD_with_t.csv'


time = []; SMD = []; Q = []
t_max = -1
for i in range(len(cases)):
    case_i = cases[i]
    folder_i = folders[case_i]
    
    df =  pd.read_csv(folder_i+file)
    
    time_i = df['t'].values
    Q_i = df['Q'].values
    SMD_i = df['SMD'].values
    
    time.append(time_i)
    Q.append(Q_i)
    SMD.append(SMD_i)
    
    if max(time_i) > t_max:
        t_max = max(time_i)


#%% plot Ql evol



plt.figure(figsize=figsize_)
# Experimental result
plt.plot([0,t_max],[Q_inj]*2, '--k', label=r'$Q_l~\mathrm{injected}$')
for i in range(len(cases)):
    case_i = cases[i]
    t_i = time[i]
    t_i = t_i / tau_x80[case_i]
    Q_i = Q[i]
    plt.plot(t_i, Q_i, formats[case_i], label=labels[case_i])

#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_time)
plt.ylabel(label_Q) 
plt.legend(loc='best', ncol=2)
#plt.legend(bbox_to_anchor=(1.0,0.75))
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
#plt.xticks(x_ticks_SMD_evol)
#plt.yticks(y_ticks_SMD_evol)
#plt.xlim(4,81)#(plot_bounds[0])
#plt.ylim(00,80)#(plot_bounds[1])
plt.savefig(folder_manuscript+'convergence_Ql.pdf')
plt.show()
plt.close()     


#%% plot SMD evol



plt.figure(figsize=figsize_)
for i in range(len(cases)):
    case_i = cases[i]
    t_i = time[i]
    SMD_i = SMD[i]
    plt.plot(t_i, SMD_i, formats[case_i], label=labels[case_i])

#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_time)
plt.ylabel(label_SMD) 
#plt.legend(bbox_to_anchor=(1.0,0.75))
plt.grid()
plt.tight_layout()
plt.yticks([10,12,14,16,18,20])
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
#plt.xticks(x_ticks_SMD_evol)
#plt.yticks(y_ticks_SMD_evol)
#plt.xlim(4,81)#(plot_bounds[0])
#plt.ylim(00,80)#(plot_bounds[1])
plt.savefig(folder_manuscript+'convergence_SMD.pdf')
plt.show()
plt.close()       
