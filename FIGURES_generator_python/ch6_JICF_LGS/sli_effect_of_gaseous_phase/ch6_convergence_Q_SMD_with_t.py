
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""





import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import pickle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
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
label_time = r'$t^{\prime}$'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'
label_Q = r'$Q_l$ [$\mathrm{mm}^3~\mathrm{s}^{-1}$]'



folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'






cases = ['APTE', 'full_no_ALM',
         'full_ALM_initial', 'full_ALM_FDC_0p10',
         'full_ALM_FDC_0p24', 'full_ALM_FDC_0p30']


folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_full_no_ALM  = folder + '/FULL_domain/LGS_no_ALM/store_variables/'
folder_full_ALM_initial = folder + '/FULL_domain/LGS_ALM_initial/store_variables/'
folder_full_FDC_0p10 = folder + '/FULL_domain/LGS_ALM_FDC_0p10/store_variables/'
folder_full_FDC_0p24 = folder + '/FULL_domain/LGS_ALM_FDC_0p24/store_variables/'
folder_full_FDC_0p30 = folder + '/FULL_domain/LGS_ALM_FDC_0p30/store_variables/'

folders = {'APTE': folder_APTE ,
           'full_no_ALM': folder_full_no_ALM,
           'full_ALM_initial': folder_full_ALM_initial,
           'full_ALM_FDC_0p10': folder_full_FDC_0p10,
           'full_ALM_FDC_0p24': folder_full_FDC_0p24,
           'full_ALM_FDC_0p30': folder_full_FDC_0p30}

SMD_cases_x_evol = [19.52, 12.30, 12.30, 13.05, 13.41, 9.98]


label_APTE = r'$\mathrm{Prescribed}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'
label_full_ALM_FDC_0p10 = r'$\mathrm{ALM~tilted}$'
label_full_ALM_FDC_0p24 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p30 = r'$\mathrm{ALM~forced}$'

labels = {'APTE': label_APTE ,
           'full_no_ALM': label_full_no_ALM,
           'full_ALM_initial': label_full_ALM_initial,
           'full_ALM_FDC_0p10': label_full_ALM_FDC_0p10,
           'full_ALM_FDC_0p24': label_full_ALM_FDC_0p24,
           'full_ALM_FDC_0p30': label_full_ALM_FDC_0p30}




formats = {'APTE':'-k', 
           'full_no_ALM':':k', 
           'full_ALM_initial':'-b',
           'full_ALM_FDC_0p10':'-r',
           'full_ALM_FDC_0p24':'-y',
           'full_ALM_FDC_0p30':'-g'}

tau_x80 = {'APTE':0.7017, 
           'full_no_ALM':0.6936, 
           'full_ALM_initial':0.6959,
           'full_ALM_FDC_0p10':0.6829,
           'full_ALM_FDC_0p24':0.6843,
           'full_ALM_FDC_0p30':0.6812}


#%% Experimental data and simulation parameters (do not touch)



params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']
Q_inj = params_simulation['Q_inj'] *1e9







#%% get data

file = 'convergence_x80mm_Q_SMD_with_t.csv'

# Q_factors for scaling
Q_factors = [1,1,1,1,1,1]
Q_factors = [1.005,1.06,1.05,1.01,1.01,1.14]


time = []; SMD = []; Q = []
t_max = -1
SMD_factors = []
for i in range(len(cases)):
    case_i = cases[i]
    folder_i = folders[case_i]
    
    df =  pd.read_csv(folder_i+file)
    
    time_i = df['t'].values
    Q_i = df['Q'].values
    SMD_i = df['SMD'].values

    print(Q_i[-1])
    
    Q_i = Q_i/Q_factors[i]
    
    if case_i == 'full_ALM_initial':
        dt = np.diff(time_i)[-1]
        # append stuff
        t_new = time_i[-1]
        Q_last = Q_i[-1]
        SMD_last = SMD_i[-1]
        while t_new < 2.1:
            t_new = t_new + dt
            time_i = np.append(time_i, t_new)
            Q_i = np.append(Q_i, Q_last)
            SMD_i = np.append(SMD_i, SMD_last)
        
        
    
    if case_i == 'full_ALM_FDC_0p10':
        index = np.where(time_i > 2)[0][0]
        for j in range(index,len(SMD_i)):
            SMD_i[j]  = SMD_i[index]
    
    time.append(time_i/tau_x80[case_i])
    Q.append(Q_i)
    SMD.append(SMD_i)
    
    if max(time_i/tau_x80[case_i]) > t_max:
        t_max = max(time_i/tau_x80[case_i])
        
    SMD_factors.append(SMD_i[-1]/SMD_cases_x_evol[i])



# EXTRA: read csv file 'full_ALM_FDC_0p30'
file_slurm_FDC_0p30 = pd.read_csv( folder_full_FDC_0p30 + 'slurm_SMD_LGS_ALM_FDC_0p30.csv')
t_0p30 = file_slurm_FDC_0p30['time'].values
t_0p30 -= t_0p30[0]
SMD_evol_FDC_0p30 = file_slurm_FDC_0p30['SMD'].values
vol_evol_FDC_0p30 = file_slurm_FDC_0p30['vol_total'].values

# calculate flux
Q_evol_FDC_0p30 = np.zeros(len(t_0p30))
for n in range(1,len(vol_evol_FDC_0p30)):
    Q_n = vol_evol_FDC_0p30[n]/t_0p30[n]
    Q_evol_FDC_0p30[n] = Q_n*1e9/Q_factors[-1]
    
t_0p30 = t_0p30*1e3/tau_x80['full_ALM_FDC_0p30']

SMD_factors[-1] = SMD_evol_FDC_0p30[-1]/SMD_cases_x_evol[-1]
SMD_evol_FDC_0p30 = SMD_evol_FDC_0p30/SMD_factors[-1] 


#%% plot Ql evol


plt.figure(figsize=figsize_)
# Experimental result
plt.plot([0,t_max],[Q_inj]*2, '--k', label=r'$Q_l~\mathrm{injected}$')
for i in range(len(cases)-1):
    case_i = cases[i]
    t_i = time[i]
    Q_i = Q[i]
    plt.plot(t_i, Q_i, formats[case_i], label=labels[case_i])

plt.plot(t_0p30, Q_evol_FDC_0p30,
         formats[cases[-1]], label=labels[cases[-1]])

#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_time)
plt.ylabel(label_Q) 
plt.legend(loc='best', ncol=2)
#plt.legend(bbox_to_anchor=(1.0,0.75))
plt.grid()
plt.tight_layout()
plt.ylim(0,8000)
plt.yticks(np.arange(0,8001,2000))
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
for i in range(len(cases)-1):
    case_i = cases[i]
    t_i = time[i]
    SMD_i = SMD[i]/SMD_factors[i]
    plt.plot(t_i, SMD_i, formats[case_i], label=labels[case_i])
    
plt.plot(t_0p30, SMD_evol_FDC_0p30,
         formats[cases[-1]], label=labels[cases[-1]])
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_time)
plt.ylabel(label_SMD) 
plt.grid()
plt.tight_layout()
#plt.yticks([10,12,14,16,18,20])
plt.yticks([0,5,10,15,20])
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
#plt.xticks(x_ticks_SMD_evol)
#plt.yticks(y_ticks_SMD_evol)
#plt.xlim(4,81)#(plot_bounds[0])
#plt.ylim(00,80)#(plot_bounds[1])
plt.savefig(folder_manuscript+'convergence_SMD.pdf')
plt.show()
plt.close()




#%%

fig, ax1 = plt.subplots(figsize=figsize_)

# data for main plot

# Experimentql result
ax1.plot([0,t_max],[Q_inj]*2, '--k', label=r'$Q_l~\mathrm{injected}$')
for i in range(len(cases)-1):
    case_i = cases[i]
    t_i = time[i]
    Q_i = Q[i]
    ax1.plot(t_i, Q_i, formats[case_i], label=labels[case_i])

ax1.plot(t_0p30, Q_evol_FDC_0p30,
         formats[cases[-1]], label=labels[cases[-1]])


# characteristics main plot
ax1.set_xlabel(label_time)
ax1.set_ylabel(label_Q)

'''
ax1.set_xlim(4,83)
ax1.set_ylim(00,85)
ax1.set_xticks(x_ticks_SMD_evol)
ax1.set_yticks(y_ticks_SMD_evol)
'''

ax1.legend(loc='upper right', ncol=2)
ax1.grid()
#ax1.grid(which='major',linestyle='-',linewidth=4*FFIG)
#ax1.grid(which='minor',linestyle='--')

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
#ip = InsetPosition(ax1, [0.45,0.40,0.3,0.3])
ip = InsetPosition(ax1, [0.25,0.10,0.6,0.3])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=1, fc="none", ec='0.5')

# data for embedded plot


ax2.plot([0,t_max],[Q_inj]*2, '--k', label=r'$Q_l~\mathrm{injected}$')
for i in range(len(cases)-1):
    case_i = cases[i]
    t_i = time[i]
    Q_i = Q[i]
    ax2.plot(t_i, Q_i, formats[case_i], label=labels[case_i])
ax2.plot(t_0p30, Q_evol_FDC_0p30,
         formats[cases[-1]], label=labels[cases[-1]])

# characteristics embedded plot
ax2.set_xlim((2.5,3.8))
ax2.set_ylim((3500,4000))
#ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 40*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')



# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_Ql.pdf')
plt.show()
plt.close()
