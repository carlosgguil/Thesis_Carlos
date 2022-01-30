# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 50*FFIG
plt.rcParams['ytick.labelsize'] = 50*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True
plt.rcParams['legend.framealpha'] = 1.0

# Main folder
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/inlet_profile/'


figsize_ = (FFIG*18,FFIG*13)

label_zc = r'$z_c~\left[\mathrm{mm} \right]$'
label_u_mean_c = r'$\overline{u}_c~\left[ m.s^{-1}\right]$'
label_TKE = r'$TKE~\left[ J.kg^{-1}\right]$'

#%% Read data



df = pd.read_csv(folder+'data_over_line_upstream_injector.csv')

x = df['Points_0'].values*1e3
y = df['Points_1'].values
z = df['Points_2'].values
u_mean_xc = df['U_MEAN_xc'].values
TKE  = df['TKE'].values

zc = x - x[0]

#%% Plot

fig = plt.figure(figsize=figsize_)
ax = fig.add_subplot(111)
lns1 = ax.plot(u_mean_xc,zc,color='k')
ax2 = ax.twiny()
lns3 = ax2.plot(TKE,zc,'b')
ax.grid()
ax.set_ylabel(label_zc)
ax.set_xlabel(label_u_mean_c)
ax2.set_xlabel(label_TKE)
#ax.set_ylim(ylims_xb_zb[i])
#ax2.set_ylim(ylims_w[i])
ax2.tick_params(axis='x', colors='blue')
ax2.xaxis.label.set_color('blue')
plt.tight_layout()
plt.savefig(folder_manuscript+'gas_inlet_profiles.pdf')
plt.show()
plt.close()

