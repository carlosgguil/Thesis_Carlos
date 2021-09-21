# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
FFIG = 0.5
import matplotlib.pyplot as plt
import numpy as np

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/'

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
plt.rcParams['lines.markersize'] = 40*FFIG


x_dx20 = [5,10,15,20]
x_dx10 = [5,10]

SMD_UG75_dx20  = [139.2, 130.7, 124.8, 121.5]
SMD_UG75_dx10  = [80.5, 69.4]

SMD_UG100_dx20 = [132.7, 120.7, 117.7, 118.6]
SMD_UG100_dx10 = [72.0, 64.6]


#Plot
plt.figure(figsize=(FFIG*22,FFIG*13))
plt.plot(x_dx20, SMD_UG75_dx20, '^-k',label=r'UG75\_DX20')
plt.plot(x_dx10, SMD_UG75_dx10, '*-k',label=r'UG75\_DX20')
plt.plot(x_dx20, SMD_UG100_dx20, '^-b',label=r'UG75\_DX20')
plt.plot(x_dx10, SMD_UG100_dx10, '*-b',label=r'UG75\_DX20')
plt.xlabel('x [mm]')
plt.ylabel(r'SMD [$\mu$m]')
plt.legend(loc='best')
plt.xticks([5,10,15,20])
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_values.pdf')
plt.savefig(folder_manuscript+'SMD_values.eps',format='eps',dpi=1000)
plt.show()
plt.close()


plt.rcParams['text.usetex'] = False