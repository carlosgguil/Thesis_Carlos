# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
FFIG = 1.0
import matplotlib.pyplot as plt
import numpy as np

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch4_SLI/'

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


sp = spray

# Characteristic time in ms
#t_char = 0.45/100 # dInj/uL
t_char = sp.time_instants[0]*1e3 # Arrival time of first droplet to plane
   
x_values = (sp.t_acc_array - sp.time_instants[0])*1e3
x_values = x_values/t_char*10
x_label = 'Accumulation time [-]'    
y_values = sp.cost_array
y_label = r'Normalized MSE (\%)'
    
#Plot
plt.figure(figsize=(FFIG*22,FFIG*13))
plt.plot(x_values, y_values*100, '-k')
plt.plot([x_values[0],x_values[-1]],[3]*2,'--k')
plt.xlabel(x_label)
plt.ylabel(y_label)
'''
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
'''
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spray_convergence.pdf')
plt.savefig(folder_manuscript+'spray_convergence.eps',format='eps',dpi=1000)
plt.show()
plt.close()


plt.rcParams['text.usetex'] = False