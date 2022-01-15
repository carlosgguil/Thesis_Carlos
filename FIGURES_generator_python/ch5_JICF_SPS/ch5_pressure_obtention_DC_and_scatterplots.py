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

figsize_ = (FFIG*22,FFIG*18)


folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/pressure_difference_mean_dense_core/data/'
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/pressure_obtention_mean_DC/'

    
    
# dibujar inyector
d_inj = 0.45;
r_inj = d_inj/2
Np    = 60
d_theta = 360/Np
x_inj = []; y_inj = [];
th_i = 0
for i in range(Np+1):
    x_i = r_inj*np.cos(th_i*np.pi/180)
    y_i = r_inj*np.sin(th_i*np.pi/180)
    x_inj.append(x_i)
    y_inj.append(y_i)
    th_i += d_theta
    
    

labelpad_ = 0
x_label_ = r'$x~[\mathrm{mm}]$'
y_label_ = r'$y~[\mathrm{mm}]$'
z_label_ = r'$z~[\mathrm{mm}]$'
p_label_ = r'$p~[\mathrm{kPa}]$'


x_lim_ = (-0.3,0.7)
y_lim_ = (-0.5,0.5)

#%% Read and retrieve data from UG100_DX10

df_wind = pd.read_csv(folder+'UG100_DX10_pressure_data_windward_surface.csv')
df_lee  = pd.read_csv(folder+'UG100_DX10_pressure_data_leeward_surface.csv')

p_mean_wind = df_wind['P_MEAN'].values/1e3
x_wind = df_wind['Points_0'].values*1e3
y_wind = df_wind['Points_1'].values*1e3
z_wind = df_wind['Points_2'].values*1e3

p_mean_lee = df_lee['P_MEAN'].values/1e3
x_lee = df_lee['Points_0'].values*1e3
y_lee = df_lee['Points_1'].values*1e3
z_lee = df_lee['Points_2'].values*1e3

p_max = max(max(p_mean_wind),max(p_mean_lee))
p_min = min(min(p_mean_wind),min(p_mean_lee))

p_lim = (-45, 30) #(p_min, p_max)
p_ticks = np.linspace(p_lim[0],p_lim[1],6)




#%% 2D plots




marker_size_   = 200*FFIG

import matplotlib.ticker as mtick


plt.figure(figsize=figsize_)
plt.title(r'$\mathrm{Windward}$')
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(x_wind, y_wind, c = p_mean_wind, facecolors='none', s=marker_size_, cmap=cm.inferno) 
plt.plot(x_inj,y_inj,'--k',alpha=0.5)
#plt.plot(0,0,'--o',markersize=100,markerstyle='--')
plt.xlabel(x_label_)
plt.ylabel(y_label_)
plt.xlim(x_lim_)
plt.grid(y_lim_)
cbar = plt.colorbar(orientation="vertical")
cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
cbar.set_label(p_label_,labelpad=labelpad_)
cbar.set_ticks(p_ticks)
plt.clim(p_lim)
plt.tight_layout()
plt.show()
plt.savefig(folder_manuscript+'p_mean_scatter_UG100_DX10_windward.png')
plt.close()



plt.figure(figsize=figsize_)
plt.title(r'$\mathrm{Leeward}$')
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(x_lee, y_lee, c = p_mean_lee, facecolors='none', s=marker_size_, cmap=cm.inferno) 
plt.plot(x_inj,y_inj,'--k',alpha=0.5)
plt.xlabel(r'$x~[\mathrm{mm}]$')
plt.ylabel(r'$y~[\mathrm{mm}]$')
plt.xlim(x_lim_)
plt.ylim(y_lim_)
plt.grid()
cbar = plt.colorbar(orientation="vertical")
cbar.set_label(p_label_,labelpad=labelpad_)
cbar.set_ticks(p_ticks)
plt.clim(p_lim)
plt.tight_layout()
plt.show()
plt.savefig(folder_manuscript+'p_mean_scatter_UG100_DX10_leeward.png')
plt.close()


