# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:17:29 2022

@author: Carlos Garcia
"""

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['lines.markersize'] =  40*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*26,FFIG*18)


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/instabilities_resolution/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/instabilities_injector_observation/data_disk_and_injector/'


label_TKE  = r'$TKE~[J~kg^{-1}]$'
label_vort = r'$\omega~[s^{-1}]$'
label_y_plus = r'$y^+$'
label_PDF = r'$\mathrm{PDF}$'#r'$\mathrm{PDF}$'
label_dx10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_dx20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'

# bars width
bars_width = 1


x_label_pad = 4*FFIG
    
n_bins = 20
z_min = -10E-3 #-0.12E-3
    
df_inj_dx10 = pd.read_csv(folder+'/UG100_DX10_data_injector.csv')
df_inj_dx20 = pd.read_csv(folder+'/UG100_DX20_data_injector.csv')

#%% Select data


y_plus_inj_dx10 = []; vort_inj_dx10 = []; TKE_inj_dx10 = []
for i in range(len(df_inj_dx10)):
    z_i = df_inj_dx10['Points_2'].values[i]
    if z_i > z_min:
        y_plus_inj_dx10.append(df_inj_dx10['Y_PLUS_MEAN'].values[i])
        vort_inj_dx10.append(df_inj_dx10['VORT_MEAN_Magnitude'].values[i])
        TKE_inj_dx10.append(0.5*df_inj_dx10['U_RMS_Magnitude'].values[i]**2)
    

y_plus_inj_dx20 = []; vort_inj_dx20 = []; TKE_inj_dx20 = []
for i in range(len(df_inj_dx20)):
    z_i = df_inj_dx20['Points_2'].values[i]
    if z_i > z_min:
        y_plus_inj_dx20.append(df_inj_dx20['Y_PLUS_MEAN'].values[i])
        vort_inj_dx20.append(df_inj_dx20['VORT_MEAN_Magnitude'].values[i])
        TKE_inj_dx20.append(0.5*df_inj_dx20['U_RMS_Magnitude'].values[i]**2)

#%% PDF of y+ 

y_plus_min = min(min(y_plus_inj_dx20),min(y_plus_inj_dx10))
y_plus_max = max(max(y_plus_inj_dx20),max(y_plus_inj_dx10))


n_dx10, bins_y_plus_dx10 = np.histogram(np.sort(y_plus_inj_dx10), n_bins, range = (y_plus_min,y_plus_max))
n_dx20, bins_y_plus_dx20 = np.histogram(np.sort(y_plus_inj_dx20), n_bins, range = (y_plus_min,y_plus_max))

d_y_plus = (y_plus_max - y_plus_min)/n_bins

hist_values_y_plus_dx10 = bins_y_plus_dx10[:-1] + d_y_plus/2
hist_values_y_plus_dx20 = bins_y_plus_dx20[:-1] + d_y_plus/2

n_bins_dx20 = int((max(bins_y_plus_dx20) - min(bins_y_plus_dx20))/d_y_plus)
n_bins_dx10 = int((max(bins_y_plus_dx10) - min(bins_y_plus_dx10))/d_y_plus)



plt.figure(figsize=figsize_) 
plt.hist(hist_values_y_plus_dx10, n_bins, weights = n_dx10, color='black', rwidth = bars_width/2, density = True, label=label_dx10)
plt.hist(hist_values_y_plus_dx20-d_y_plus/2, n_bins, weights = n_dx20, color='grey', rwidth = bars_width/2, density = True, label=label_dx20) 
plt.plot([5]*2,[0,100000],'--k',label=r'$y^+ = 5$',zorder=0)
plt.xlabel(label_y_plus, labelpad = x_label_pad)
plt.ylabel(label_PDF)
plt.yscale('log')
plt.yticks([1e-4,10e-4,10e-3,10e-2,10e-1])
plt.ylim(1e-4, 1e0)
#ax.yaxis.set_ticklabels([])
plt.grid(axis='y', alpha=0.75)
#plt.title(titles_xplanes[i])
plt.legend()
plt.tight_layout()
plt.savefig(folder_manuscript+'y_plus_injector.pdf')
plt.show()
plt.close()

#%% PDF of vorticity magnitude


vort_min = min(min(vort_inj_dx20),min(vort_inj_dx10))
vort_max = max(max(vort_inj_dx20),max(vort_inj_dx10))


n_dx10, bins_vort_dx10 = np.histogram(np.sort(vort_inj_dx10), n_bins, range = (vort_min,vort_max))
n_dx20, bins_vort_dx20 = np.histogram(np.sort(vort_inj_dx20), n_bins, range = (vort_min,vort_max))

d_vort = (vort_max - vort_min)/n_bins

hist_values_vort_dx10 = bins_vort_dx10[:-1] + d_vort/2
hist_values_vort_dx20 = bins_vort_dx20[:-1] + d_vort/2

n_bins_dx20 = int((max(bins_vort_dx20) - min(bins_vort_dx20))/d_vort)
n_bins_dx10 = int((max(bins_vort_dx10) - min(bins_vort_dx10))/d_vort)



plt.figure(figsize=figsize_) 
plt.hist(hist_values_vort_dx10, n_bins, weights = n_dx10, color='black', rwidth = bars_width/2, density = True, label=label_dx10)
plt.hist(hist_values_vort_dx20-d_vort/2, n_bins, weights = n_dx20, color='grey', rwidth = bars_width/2, density = True, label=label_dx20) 
plt.xlabel(label_vort, labelpad = x_label_pad)
plt.ylabel(label_PDF)
plt.yscale('log')
#plt.yticks([10e-4,10e-3,10e-2,10e-1,10e0])
#plt.ylim(1e-4, 30e0)
#ax.yaxis.set_ticklabels([])
plt.grid(axis='y', alpha=0.75)
#plt.title(titles_xplanes[i])
plt.legend()
plt.tight_layout()
plt.show()
plt.close()




#%% Scatterplot y+ vs vorticity

marker_size_ = 50

plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(y_plus_inj_dx20, vort_inj_dx20, s=marker_size_, label=label_dx20) 
plt.scatter(y_plus_inj_dx10, vort_inj_dx10, s=marker_size_, facecolors = 'none', edgecolors='k',label=label_dx10) 
plt.xlabel(label_y_plus)
plt.ylabel(label_vort)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig(folder_manuscript+'p_mean_scatter_UG100_DX10_leeward.png')
plt.close()
