# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:38:38 2021

@author: d601630
"""

def calc_SMD(diam):
    diam = np.array(diam)
    SMD  = np.sum(diam**3)/np.sum(diam**2)
    
    return SMD


from math import floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Change size of figures 
FFIG = 0.5

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 70*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 70*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 70*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 20*FFIG
plt.rcParams['axes.titlesize']  = 70*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 30*FFIG
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams['text.usetex'] = True
#rc('text.latex', preamble='\usepackage{color}')




folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch9_lagrangian/'
folder_simus = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/LGS_simus/data_droplets_BIMER_LGS/'


folder = './'

# Maps N levels
N_LEVELS = 51

# threshold on droplet number
n_dr_th = 0

# threshold on flux
Q_th = 0.1   

# droplet threshold diameter to distinguish pilot and take-off
D_th = 17

# Map coordinates
x_min = 10
x_max = 35
y_min = -43.5
y_max = 46.5

DX = x_max - x_min
DY = y_max - y_min

# Define figsize
AR = DX/DY
figsize_ = (FFIG*18*AR,FFIG*16)
figsize_maps_subplots = (5*FFIG*16*AR,FFIG*16)
figsize_bar = (FFIG*50,FFIG*20)

pad_title_maps = 30

# labels
x_label_ = r'$x~[\mathrm{mm}]$'
y_label_ = r'$y~[\mathrm{mm}]$'

label_SMD = r'$SMD~[\mu \mathrm{m}]$'
label_u_axial = r'$u~[\mathrm{m}~s^{-1}]$'
label_u_vertical = r'$v~[\mathrm{m}~s^{-1}]$'
label_flux = r'$Flux$'


# grid limits
x_min = 10
x_max = 35
y_min = -43
y_max = 46

dx = 1
dy = 1

#x_grid = np.arange(x_min-dx/2,x_max+dx/2,dx)
#y_grid = np.arange(y_min-dy/2,y_max+dx/2,dy)
x_grid = np.arange(x_min,x_max+dx/100,dx)
y_grid = np.arange(y_min,y_max+dy/100,dy)

xx, yy = np.meshgrid(x_grid, y_grid)

nx = len(x_grid)
ny = len(y_grid)

# levels calculation
levels_map_SMD = np.linspace(9,34,51)
levels_map_u_axial = np.linspace(-23,53,77)
levels_map_u_vertical = np.linspace(-17,17,34)

#%% read cases

cases_labels = ['Baseline','Evap','ALM']

cases = ['baseline_no_ALM_no_evap',
         'with_ALM_no_evap',
         'no_ALM_with_evap']
dt_cases = [8.6799802*1e-3,
            10.39020215*1e-3,
            5.279644288*1e-3]

d_values_cases = []
for k in range(len(cases)):
    case = cases[k]
    dt = dt_cases[k]
    
    df_no_grid = pd.read_csv(folder_simus+case+'/lgs_droplets_within_region_no_grid.csv')
    
    
    times_all = df_no_grid[' time'].values
    t = np.unique(times_all)
    t_acq = t[::10]
    
    
    global_indices_all =  df_no_grid['global_index'].values
    x_values_all = df_no_grid['xp'].values*1e3
    y_values_all = df_no_grid['yp'].values*1e3
    z_values_all = df_no_grid['zp'].values*1e3
    d_values_all = df_no_grid['diameter'].values*1e6
    u_values_all = df_no_grid['u'].values
    v_values_all = df_no_grid['v'].values
    w_values_all = df_no_grid['w'].values
    
    z_th = 0.5
    
    # filter droplets and get indices vectors
    xp_indices = []; yp_indices = []
    times = []
    global_indices =  []
    x_values = []; y_values = []; d_values  = []
    u_values = []; v_values = []; w_values  = []
    for i in range(len(times_all)):
        '''
        if times_all[i] not in t_acq:
            continue
        '''
        
        x_i = x_values_all[i]
        y_i = y_values_all[i]
        z_i = z_values_all[i]
        d_i = d_values_all[i]
        gi_i = global_indices_all[i]
        
        
        # check if droplet is outside 2D bounds
        if (x_i < x_min) or (x_i > x_max):
            continue
        if (y_i < y_min) or (y_i > y_max):
            continue
        
        '''
        # filter within z-slice of thickness 2*z_th
        if (z_i < -1*z_th) or (z_i > z_th):
            continue
        '''
        
        '''
        # ignore if same droplet has already been sampled once
        if gi_i in global_indices:
            continue
        '''
        
        d_values.append(d_i)
        
    d_values_cases.append(np.array(d_values))
        

#%% plot histogram
figsize_bar = (FFIG*25,FFIG*15)


d_min = 0.5
d_max = 40
n_bins = 20
dd = (d_max - d_min)/n_bins

f0_baseline, bins_baseline  = np.histogram(np.sort(d_values_cases[0]), n_bins, 
                                          range=(d_min,d_max), density=True)
f0_ALM, bins_ALM  = np.histogram(np.sort(d_values_cases[1]), n_bins, 
                                          range=(d_min,d_max), density=True)
f0_evap, bins_evap  = np.histogram(np.sort(d_values_cases[2]), n_bins, 
                                          range=(d_min,d_max), density=True)
bins_ = bins_baseline
medium_values = (bins_[1:] + bins_[:-1])*0.5

bars_width = dd



linewidth_cotas = 5*FFIG
linewidth_arrow = 5*FFIG
head_width_ = 0.02
head_length_ = 0.7
x_arrows = 25
y_arrows = 0.40
l_arrows = 4

plt.figure(figsize=figsize_bar)
#plt.title('Filming mean $Q_l$')
plt.bar(medium_values+dd, f0_baseline,  width=bars_width, color='black', label=r'$\mathrm{Baseline}$', edgecolor='white')
plt.bar(medium_values, f0_evap,  alpha=0.75, width=bars_width, color='grey', label=r'$\mathrm{Evap}$', edgecolor='white')
#plt.bar(medium_values+dd/2, f0_ALM,  width=bars_width, color='blue', edgecolor='white')
#plt.bar(r1, Q_x_mean_xD_05p00, yerr=Q_x_RMS_xD_05p00, width=barWidth, color='grey', edgecolor='white', label=label_xD_05p00, capsize=barWidth*20)
#plt.bar(r1+0.25, Q_x_mean_xD_06p67, yerr=Q_x_RMS_xD_06p67, width=barWidth, color='red', edgecolor='white', label=label_xD_06p67, capsize=barWidth*20)
plt.xlabel(r'$D~[\mu \mathrm{m}]$')#, fontweight='bold')
plt.arrow(x_arrows, y_arrows, l_arrows, 0, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
plt.arrow(x_arrows, y_arrows, -l_arrows, 0, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
plt.plot([x_arrows]*2, [0,0.45], '--k', linewidth = linewidth_cotas)
plt.text(x_arrows+l_arrows/2, y_arrows-0.04, r'$\mathrm{Take-off}$')
plt.text(x_arrows-l_arrows*1.2, y_arrows-0.04, r'$\mathrm{Pilot}$')
'''
plt.arrow(x_arrow, y_arrow, 0, l_arrow, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
plt.text(x_arrow-0.08, y_arrow-l_arrow/2, r'$\mathrm{Cost~\\savings}$',
             color='black', rotation='vertical',fontsize=90*FFIG)
'''
plt.ylabel(r'$f_0~\left( D\right)$')
plt.ylim(0,0.45)
plt.legend(loc='upper left')
plt.tight_layout()
plt.yticks([0,0.1,0.2,0.3,0.4])
plt.savefig(folder_manuscript+'droplets_histograms.pdf')
plt.show()
plt.close()


