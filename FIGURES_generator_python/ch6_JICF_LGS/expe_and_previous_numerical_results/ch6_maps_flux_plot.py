
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
from sprPost_calculations import get_discrete_spray, get_sprays_list
import sprPost_plot as sprPlot

def plot_grid_func(y_values, z_values):
    Ny = len(y_values)
    Nz = len(z_values)
    grid_y_values = np.linspace(y_values[0]-1, y_values[-1]+1,len(y_values)+1)
    grid_z_values = np.linspace(z_values[0]-1, z_values[-1]+1,len(z_values)+1)
    y_min = min(grid_y_values); y_max = max(grid_y_values)
    z_min = min(grid_z_values); z_max = max(grid_z_values)
    # Plot vertical lines, so we need to iterate between z lines
    for i in range(0,Ny+1):
        plt.plot([grid_y_values[i]]*2,[z_min, z_max], 
                 linewidth=2, color='k')
    # Plot now horizontal lines, so we need to iterate between z lines
    for j in range(0,Nz+1):
        plt.plot([y_min, y_max], [grid_z_values[j]]*2, 
                 linewidth=2, color='k')                   
    

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/previous_numerical_results/'
folder_numerics = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/LGS_previous_numerical_works/maps_data/'
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'

# Change size of figures 
FFIG = 0.5
#mpl.rcParams['font.size'] = 40*fPic
plt.rcParams['xtick.labelsize']  = 50*FFIG
plt.rcParams['ytick.labelsize']  = 50*FFIG
plt.rcParams['axes.labelsize']   = 50*FFIG
plt.rcParams['axes.labelpad']    = 30*FFIG
plt.rcParams['axes.titlesize']   = 50*FFIG
plt.rcParams['legend.fontsize']  = 30*FFIG
plt.rcParams['lines.linewidth']  = 5*FFIG
plt.rcParams['lines.markersize'] = 20*FFIG

plt.rcParams['xtick.labelsize'] = 80*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 80*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 30*FFIG


plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
AR = 1#(35-10)/(46+43)
figsize_expe = (AR*FFIG*16,FFIG*15)  
figsize_ = (AR*FFIG*15,FFIG*15)    
figsize_senoner = (AR*FFIG*15,FFIG*15.4)    
figsize_eckel = (AR*FFIG*18.3,FFIG*15)            


#%% plot parameters


label_ql = r'$q_l~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'

levels_ = np.linspace(0,5.5,12)

x_lim_ = (-12, 12)
y_lim_  = (0,23)
y_ticks_ = range(0, 23, 4)

PLOT_GRID_EXPE = True

x_label_ = '$y ~ [\mathrm{mm}]$'
y_label_ = '$z ~ [\mathrm{mm}]$'

dpi_ = 120

#%% Recover data to plot

# expes
with open(folder_expe+'/pickle_map_expe_high_we', 'rb') as f:
    obj = pickle.load(f)
    expe_y_values = obj[0]
    expe_z_values = obj[1]
    expe_yy_values = obj[2]
    expe_zz_values = obj[3]
    expe_flux_values = obj[4]

# jaegle
with open(folder_numerics+'/pickle_map_jaegle', 'rb') as f:
    obj = pickle.load(f)
    jaegle_y_values = obj[0]
    jaegle_z_values = obj[1]
    jaegle_yy_values = obj[2]
    jaegle_zz_values = obj[3]
    jaegle_flux_values = obj[4]


# senoner
with open(folder_numerics+'/pickle_map_senoner', 'rb') as f:
    obj = pickle.load(f)
    senoner_y_values = obj[0]
    senoner_z_values = obj[1]
    senoner_yy_values = obj[2]
    senoner_zz_values = obj[3]
    senoner_flux_values = obj[4]

# eckel
with open(folder_numerics+'/pickle_map_eckel', 'rb') as f:
    obj = pickle.load(f)
    eckel_y_values = obj[0]
    eckel_z_values = obj[1]
    eckel_yy_values = obj[2]
    eckel_zz_values = obj[3]
    eckel_flux_values = obj[4]

#%% Plots


# expe
plt.figure(figsize=figsize_expe)
plt.contourf(expe_yy_values, expe_zz_values, expe_flux_values, levels = levels_, cmap='binary')
#plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(expe_yy_values, expe_zz_values, expe_flux_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
if PLOT_GRID_EXPE:
    plot_grid_func(expe_y_values, expe_z_values)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel(x_label_) 
plt.ylabel(y_label_) 
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Experimental}$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(x_lim_)#(plot_bounds[0])
plt.ylim(y_lim_)#(plot_bounds[1])
plt.yticks(y_ticks_)
plt.savefig(folder_manuscript+'map_expe.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()             

# Jaegle
plt.figure(figsize=figsize_)
plt.contourf(jaegle_yy_values, jaegle_zz_values, jaegle_flux_values, levels = levels_, cmap='binary')
#plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(jaegle_yy_values, jaegle_zz_values, jaegle_flux_values, 
                      levels = levels_[1:], colors= 'k', linewidths = 2*FFIG)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel(x_label_) 
#plt.ylabel(y_label_) 
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(x_lim_)#(plot_bounds[0])
plt.ylim(y_lim_)#(plot_bounds[1])
plt.yticks([])
plt.savefig(folder_manuscript+'map_jaegle.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()             


# senoner
plt.figure(figsize=figsize_senoner)
plt.contourf(senoner_yy_values, senoner_zz_values, senoner_flux_values, levels = levels_, cmap='binary')
#plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(senoner_yy_values, senoner_zz_values, senoner_flux_values, 
                      levels = levels_[1:], colors= 'k', linewidths = 2*FFIG)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel(x_label_) 
#plt.ylabel(y_label_) 
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Senoner~(2010)}$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(x_lim_)#(plot_bounds[0])
plt.ylim(y_lim_)#(plot_bounds[1])
plt.yticks([])
plt.savefig(folder_manuscript+'map_senoner.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()             

# eckel
plt.figure(figsize=figsize_eckel)
plt.contourf(eckel_yy_values, eckel_zz_values, eckel_flux_values, levels = levels_, cmap='binary')
plt.colorbar(format = '$%.1f$',ticks=np.linspace(0,5,6), label = label_ql)
contour = plt.contour(eckel_yy_values, eckel_zz_values, eckel_flux_values, 
                      levels = levels_[1:], colors= 'k', linewidths = 2*FFIG)
plt.xlabel(x_label_) 
#plt.ylabel(y_label_) 
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Eckel ~(2016)}$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(x_lim_)#(plot_bounds[0])
plt.ylim(y_lim_)#(plot_bounds[1])
plt.yticks([])
plt.savefig(folder_manuscript+'map_eckel.png',bbox_inches='tight',dpi=dpi_)
plt.show()
plt.close()             
   

#%% Standalone colormap q

'''

a = np.array([[levels_[0],levels_[-1]]])
plt.figure(figsize=(FFIG*1.5, FFIG*18))
#plt.figure(figsize=(fPic*1.5, fPic*18))
img = plt.imshow(a, cmap="binary")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation="vertical", cax=cax, format = '$%.1f$')
cbar.set_label(label_ql,labelpad=10)
cbar.set_ticks(levels_)
#plt.tight_layout()
#plt.savefig(folder_manuscript+'scatterplots_colorbar_y.png',bbox_inches="tight")

'''