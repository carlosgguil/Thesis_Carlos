# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('../..')
from sli_functions import load_all_BIMER_global_sprays

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/histograms_size_volume/'
FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 80*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 30*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
figsize_ = (FFIG*15,FFIG*12)
figsize_no_axe = (FFIG*11.8,FFIG*12)



#%% Load sprays

# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    
sampling_planes = ['xD_03p33','xD_05p00','xD_06p67',
                   'xD_08p33','xD_10p00']
# Load sprays
sp1, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation, sampling_planes = sampling_planes)

sprays_list_all = [sp1, sp2, sp3]




#%% Parameters

# bars width
bars_width = 1

x_label_pad = 4*FFIG
label_numb_hist = r'$f_0$' #'Number hist.'
label_vol_hist  = r'$f_3$' #'Volume hist.'

# to save graphs
x_planes = ['xD03p33', 'xD05p00', 'xD06p67', 'xD08p33', 'xD10p00']
cases    = ['DX07' , 'DX10' ,'DX15' ]

# titles to plot
titles_xplanes = [r'$x_c/d_\mathrm{inj} = 3.33$', 
                  r'$x_c/d_\mathrm{inj} = 5$', 
                  r'$x_c/d_\mathrm{inj} = 6.67$', 
                  r'$x_c/d_\mathrm{inj} = 8.33$']

titles_xplanes = [r'$x_c = 1~\mathrm{mm}$', 
                  r'$x_c = 1.5~\mathrm{mm}$',  
                  r'$x_c = 2~\mathrm{mm}$', 
                  r'$x_c = 2.5~\mathrm{mm}$',  
                  r'$x_c = 3~\mathrm{mm}$']

x_label_histos = r'$\mathrm{Diameter}~[\mu \mathrm{m}]$'
y_label_histos= r'$f_0,~f_3$'

# Minimum diameters for plotting
D_min = 0
p_min = 0

# Maximum diameters for plotting
D_max_DX15 = 100
D_max_DX10 = 100
D_max_DX07 = 100
D_max_all = [D_max_DX07, D_max_DX10, D_max_DX15]
#D_max_all = [350]*5

# Maximum probabilities for plotting
p_max_DX15 = 0.08
p_max_DX10 = 0.14
p_max_DX07 = 0.16
p_max_all = [p_max_DX07, p_max_DX10, p_max_DX15]

# y_ticks for plotting

y_ticks_DX07 = [0, 0.05, 0.1, 0.15]
y_ticks_DX10 = [0, 0.05, 0.1, 0.15]
y_ticks_DX15 = [0, 0.02, 0.04, 0.06, 0.08]
y_ticks_all = [y_ticks_DX07 , y_ticks_DX10 , y_ticks_DX15]


# resolutions
resolutions = [7.5, 10, 15]

dmin_factor = 2

for c in range(len(cases)):

    case = cases[c]
    sprays_list = sprays_list_all[c]
    
    if case == 'DX07':
        dmin_factor = 4
    elif case == 'DX10':
        dmin_factor = 3
    elif case == 'DX15':
        dmin_factor = 2
    
    
    # xD = 03.33
    i = 0
    sp = sprays_list[i]
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    
    plt.figure(figsize=figsize_)
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label=label_numb_hist)
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label=label_vol_hist) 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
    plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}'+f' = {resolutions[c]}~\mu m$')
    plt.plot([30]*2,[0,1],'-.k',label='$30~\mu m$')
    #plt.plot([dmin_factor*resolutions[c]]*2,[0,1],'-.k',label=str(dmin_factor)+r'$\Delta x_\mathrm{min}$')
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
    
    
    # xD = 05.00
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
    plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}'+f' = {resolutions[c]}~\mu m$')
    plt.plot([30]*2,[0,1],'-.k',label='$30~\mu m$')
    #plt.plot([dmin_factor*resolutions[c]]*2,[0,1],'-.k',label=str(dmin_factor)+r'$\Delta x_\mathrm{min}$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    plt.ylabel(y_label_histos)
    plt.yticks(y_ticks_all[c])
    #ax.yaxis.set_ticklabels([])
    plt.grid(axis='y', alpha=0.75)
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    
    # xD = 06.67
    i = 2
    sp = sprays_list[i] 
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    

    
    plt.figure(figsize=figsize_no_axe) 
    ax = plt.gca()
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label=label_numb_hist)
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label=label_vol_hist) 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
    plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}'+f' = {resolutions[c]}~\mu m$')
    plt.plot([30]*2,[0,1],'-.k',label='$30~\mu m$')
    #plt.plot([dmin_factor*resolutions[c]]*2,[0,1],'-.k',label=str(dmin_factor)+r'$\Delta x_\mathrm{min}$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    #plt.ylabel(y_label_histos)
    #plt.yticks(y_ticks_all[c])
    ax.yaxis.set_ticklabels([])
    plt.grid(axis='y', alpha=0.75)
    #plt.legend(loc='best')
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    
    # xD = 08p33
    i = 3
    sp = sprays_list[i] 
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    

    
    plt.figure(figsize=figsize_no_axe) 
    ax = plt.gca()
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label=label_numb_hist)
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label=label_vol_hist) 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (fit)')
    plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}'+f' = {resolutions[c]}~\mu m$')
    plt.plot([30]*2,[0,1],'-.k',label='$30~\mu m$')
    #plt.plot([dmin_factor*resolutions[c]]*2,[0,1],'-.k',label=str(dmin_factor)+r'$\Delta x_\mathrm{min}$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    #plt.ylabel(y_label_histos)
    #plt.yticks(y_ticks_all[c])
    ax.yaxis.set_ticklabels([])
    plt.grid(axis='y', alpha=0.75)
    #plt.legend(loc='best')
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    
    # xD = 10p00
    i = 4
    sp = sprays_list[i] 
    data = sp.diam.values
    n, bins       = np.histogram(np.sort(data), sp.n_bins)
    dD = np.diff(bins)[0]
    medium_values = (bins[1:] + bins[:-1])*0.5
    vol_total = medium_values**3*n  
    

    
    plt.figure(figsize=figsize_no_axe) 
    ax = plt.gca()
    plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = bars_width/2, density = True, label=label_numb_hist)
    plt.hist(bins[:-1]+dD/2, sp.n_bins, color='grey', weights = vol_total, rwidth = 0.9/2, density = True, label=label_vol_hist) 
    plt.plot(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Logn. (corr.)')
    plt.plot(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Logn. (fit)')
    #plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}'+f' = {resolutions[c]}~\mu m$')
    plt.plot([resolutions[c]]*2,[0,1],':k',label=r'$\Delta x_\mathrm{min}$')
    plt.plot([30]*2,[0,1],'-.k',label='$30~\mu m$')
    #plt.plot([dmin_factor*resolutions[c]]*2,[0,1],'-.k',label=str(dmin_factor)+r'$\Delta x_\mathrm{min}$')
    plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
    plt.xlim([D_min,D_max_all[c]])
    plt.ylim(p_min, p_max_all[c])
    plt.xlabel(x_label_histos, labelpad = x_label_pad)
    #plt.ylabel(y_label_histos)
    #plt.yticks(y_ticks_all[c])
    ax.yaxis.set_ticklabels([])
    plt.grid(axis='y', alpha=0.75)
    plt.legend(loc='best')
    plt.title(titles_xplanes[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_'+x_planes[i]+'_histograms.pdf')
    plt.show()
    plt.close()
    
    

