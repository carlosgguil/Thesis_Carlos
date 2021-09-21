# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
FFIG = 0.5
figsize_ = (FFIG*22,FFIG*13)
import matplotlib.pyplot as plt
import numpy as np

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/spray_distributions/'

plt.rcParams['xtick.labelsize'] = 60*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

case = 'uG100_dx20'
resolution = 20

D_min = 0
D_max = 500
p_min = 0
p_max = 0.035

#%% x = 05 mm

sp = sprays_list[0][0]

data = sp.diam.values
n, bins       = np.histogram(np.sort(data), sp.n_bins)
medium_values = (bins[1:] + bins[:-1])*0.5
vol_total = medium_values**3*n  
plt.figure(figsize=figsize_)
plt.hist(bins[:-1], sp.n_bins, color='black', weights = vol_total, rwidth = 0.9, density = True, label='Histogram') 
plt.plot([resolution]*2,[0,1],':k',label=f'$\Delta x = {resolution} \mu m$')
plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
plt.xlim([D_min,D_max])
plt.ylim([p_min,p_max])
'''
plt.loglog(sp.spaceDiam, sp.lognormal.PDF_f0, color='red', label='Lognormal')
plt.loglog(sp.spaceDiam, sp.lognormal_opt.PDF_f0, color='blue', label='Lognormal (opt)')
plt.xlim([1e1,1e3])
plt.ylim([1e-4,1e-1])
'''
plt.xlabel(r'Diameter [$\mu$m]')
plt.ylabel(r'$f_3$')
plt.grid(axis='y', alpha=0.75)
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_x05_volume_histogram.pdf')
plt.savefig(folder_manuscript+case+'_x05_volume_histogram.eps',format='eps',dpi=1000)
plt.show()
plt.close()


#%% x = 10 mm

sp = sprays_list[0][1]

data = sp.diam.values
plt.figure(figsize=figsize_)
plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = 0.9, density = True, label='Histogram')
plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
plt.plot([resolution]*2,[0,1],':k',label=f'$\Delta x = {resolution} \mu m$')
plt.xlim([D_min,D_max])
plt.ylim([p_min,p_max])
plt.xlabel(r'Diameter [$\mu$m]')
plt.ylabel(r'$f_3$')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_x10_volume_histogram.pdf')
plt.savefig(folder_manuscript+case+'_x10_volume_histogram.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% x = 15 mm

sp = sprays_list[0][2]

data = sp.diam.values
plt.figure(figsize=figsize_)
plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = 0.9, density = True, label='Histogram')
plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
plt.plot([resolution]*2,[0,1],':k',label=f'$\Delta x = {resolution} \mu m$')
plt.xlim([D_min,D_max])
plt.ylim([p_min,p_max])
plt.xlabel(r'Diameter [$\mu$m]')
plt.ylabel(r'$f_3$')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_x15_volume_histogram.pdf')
plt.savefig(folder_manuscript+case+'_x15_volume_histogram.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% x = 20 mm

sp = sprays_list[0][3]

data = sp.diam.values
plt.figure(figsize=figsize_)
plt.hist(np.sort(data), sp.n_bins, color='black', rwidth = 0.9, density = True, label='Histogram')
plt.plot([sp.SMD]*2,[0,1],'--k',label='SMD')
plt.plot([resolution]*2,[0,1],':k',label=f'$\Delta x = {resolution} \mu m$')
plt.xlim([D_min,D_max])
plt.ylim([p_min,p_max])
plt.xlabel(r'Diameter [$\mu$m]')
plt.ylabel(r'$f_3$')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_x20_volume_histogram.pdf')
plt.savefig(folder_manuscript+case+'_x20_volume_histogram.eps',format='eps',dpi=1000)
plt.show()
plt.close()

plt.rcParams['text.usetex'] = False