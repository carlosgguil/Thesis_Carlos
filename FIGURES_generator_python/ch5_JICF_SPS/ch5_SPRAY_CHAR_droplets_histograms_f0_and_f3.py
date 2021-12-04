# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/spray_distributions/'
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/histograms_size_volume/'

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 80*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
figsize_ = (FFIG*19,FFIG*12)



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

sprays_list_UG75_DX10 = sp1
sprays_list_UG75_DX20 = sp2
sprays_list_UG100_DX10 = sp3
sprays_list_UG100_DX20 = sp4
sprays_list_UG100_DX20_NT = sp5

#%% Parameters

# bars width
bars_width = 1

x_label_pad = 4*FFIG
label_numb_hist = r'$f_0$' #'Number hist.'
label_vol_hist  = r'$f_3$' #'Volume hist.'

# to save graphs
x_planes = ['x05', 'x10', 'x15']
cases    = ['UG75_DX10' , 'UG75_DX20' ,
            'UG100_DX10', 'UG100_DX20', 'UG100_DX20_NT']

# titles to plot
titles_xplanes = [r'$x = 5~\mathrm{mm}$', 
                  r'$x = 10~\mathrm{mm}$', 
                  r'$x = 15~\mathrm{mm}$']


x_label_histos = r'$\mathrm{Diameter}~[\mu \mathrm{m}]$'
y_label_histos= r'$f_0,~f_3$'

# Minimum diameters for plotting
D_min = 0
p_min = 0

# Maximum diameters for plotting
D_max_UG75_DX20 = 400
D_max_UG75_DX10 = 400
D_max_UG100_DX10 = 400
D_max_UG100_DX20 = 400
D_max_UG100_DX20_NT = 400
D_max_all = [D_max_UG75_DX20, D_max_UG75_DX10,
             D_max_UG100_DX10, D_max_UG100_DX20, D_max_UG100_DX20_NT]
D_max_all = [350]*5

# Maximum probabilities for plotting
p_max_UG75_DX10 = 0.05
p_max_UG75_DX20 = 0.035
p_max_UG100_DX10 = 0.05
p_max_UG100_DX20 = 0.04
p_max_UG100_DX20_NT = 0.04
p_max_all = [p_max_UG75_DX10, p_max_UG75_DX20,
             p_max_UG100_DX10, p_max_UG100_DX20, p_max_UG100_DX20_NT]

# y_ticks for plotting
y_ticks_UG75_DX10 = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
y_ticks_UG75_DX20 = [0, 0.01, 0.02, 0.03]
y_ticks_UG100_DX10 = [0, 0.01, 0.02, 0.03, 0.04]
y_ticks_UG100_DX20 = [0, 0.01, 0.02, 0.03, 0.04]
y_ticks_UG100_DX20_NT = [0, 0.01, 0.02, 0.03, 0.04]
y_ticks_all = [y_ticks_UG75_DX10 , y_ticks_UG75_DX20 ,
               y_ticks_UG100_DX10, y_ticks_UG100_DX20, y_ticks_UG100_DX20_NT]

# resolutions
resolutions = [10, 20, 10, 20, 20]


for c in range(len(cases)):

    case = cases[c]
    sprays_list = sprays_list_all[c]
    
    
    # x = 05 mm
    i = 0
    sp = sprays_list[i]
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    
    plt.figure(figsize=figsize_)
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label='Number hist.')
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label='Volume hist.') 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
    plt.plot([resolutions[c]]*2,[0,1],':k',label=f'$\Delta x = {resolutions[c]}~\mu m$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.yticks(y_ticks_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    plt.ylabel(y_label_histos)
    plt.grid(axis='y', alpha=0.75)
    plt.legend(loc='best')
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    
    # x = 10 mm
    i = 1
    sp = sprays_list[i] 
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    

    
    plt.figure(figsize=figsize_) 
    ax = plt.gca()
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label=label_numb_hist)
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label=label_vol_hist) 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
    plt.plot([20]*2,[0,1],':k',label=r'$\Delta x = 20~\mu m$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    #plt.ylabel(y_label_histos)
    plt.yticks(y_ticks_all[c])
    ax.yaxis.set_ticklabels([])
    plt.grid(axis='y', alpha=0.75)
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    # x = 15 mm (only if UG75_DX20 or UG100_DX20)
    if c == 1 or c == 3:
        i = 2
        sp = sprays_list[i]
        data = sp.diam.values
        n, bins       = np.histogram(np.sort(data), sp.n_bins)
        dD = np.diff(bins)[0]
        medium_values = (bins[1:] + bins[:-1])*0.5
        vol_total = medium_values**3*n  
        
        plt.figure(figsize=figsize_)
        ax = plt.gca()
        plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label='Number hist.')
        plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label='Volume hist.') 
        plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
        plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
        plt.plot([20]*2,[0,1],':k',label=r'$\Delta x = 20~\mu m$')
        plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
        plt.xlim([D_min,D_max_all[c]])
        plt.ylim(p_min, p_max_all[c])
        plt.xlabel(x_label_histos, labelpad = x_label_pad)
        #plt.ylabel(y_label_histos)
        plt.yticks(y_ticks_all[c])
        ax.yaxis.set_ticklabels([])
        plt.grid(axis='y', alpha=0.75)
        plt.title(titles_xplanes[i])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
        plt.show()
        plt.close()

