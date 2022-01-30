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
figsize_ = (FFIG*25,FFIG*16)


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/instabilities_resolution/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/instabilities_injector_observation/data_lines_outside_injector/'



label_s  = '$s ~[\mathrm{mm}]$'
#label_u  = r'$u_z ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_u  = r'$w ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_dx10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_dx20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_closer = r'$z = 0.05~\mathrm{mm}$' #r'$\mathrm{Closer~injector}$'
label_further = r'$z = 0.2~\mathrm{mm}$' #r'$\mathrm{Further~injector}$'

s_lim = (-0.5,0.2)
y_lim_ = (-36,30)


df_dx10_closer_injector =  pd.read_csv(folder+'UG100_DX10_data_line_closer_injector.csv')
df_dx10_further_injector = pd.read_csv(folder+'UG100_DX10_data_line_further_injector.csv')
df_dx20_closer_injector =  pd.read_csv(folder+'UG100_DX20_data_line_closer_injector.csv')
df_dx20_further_injector = pd.read_csv(folder+'UG100_DX20_data_line_further_injector.csv')

dfs = [df_dx10_closer_injector, df_dx20_closer_injector,
       df_dx10_further_injector, df_dx20_further_injector]


# get data
x = []; uz = []; phi = []; uz_mean = []; phi_mean = []; band = []
for i in range(len(dfs)):
    df = dfs[i]
    
    x_i    = df['Points_0'].values*1e3
    uz_i   = df['U_2'].values
    phi_i  = df['LS_PHI'].values
    uz_mean_i   = df['U_MEAN_2'].values
    phi_mean_i  = df['LS_PHI_MEAN'].values
    band_i = df['LS_BAND'].values
    
    # find  interface location and flag nodes belonging to band
    FOUND_INTERFACE = False; j = 1
    while (not FOUND_INTERFACE):
        phi_i_j = phi_i[j]
        #phi_i_j = phi_mean_i[j]
        if phi_i_j < 0.5:
            j += 1
        else:
            x_int = x_i[j]
            FOUND_INTERFACE = True

    x.append(x_i - x_int)
    uz.append(uz_i)
    phi.append(phi_i)
    uz_mean.append(uz_mean_i)
    phi_mean.append(phi_mean_i)
    band.append(band_i)
    


# Closer to injector
plt.figure(figsize=figsize_)
plt.title(label_closer)
i = 0; plt.plot(x[i],uz[i],'g',label=label_dx10)
i = 1; plt.plot(x[i],uz[i],'b',label=label_dx20)
plt.plot([0]*2,[-40,40],'k')
plt.text(-0.08,-34.5,r'$\mathrm{Gas}$',fontsize=75*FFIG)
plt.text(0.01,-34.5,r'$\mathrm{Liquid}$',fontsize=75*FFIG)
#plt.yticks(y_ticks_u_vs_x)
plt.xlim(s_lim)
plt.ylim(y_lim_)
plt.xlabel(label_s)
plt.ylabel(label_u)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_data_outside_injector_z_low.pdf')
plt.show()
plt.close()

# Further to injector
plt.figure(figsize=figsize_)
plt.title(label_further)
i = 2; plt.plot(x[i],uz[i],'g',label=label_dx10)
i = 3; plt.plot(x[i],uz[i],'b',label=label_dx20)
plt.plot([0]*2,[-40,40],'k')
plt.text(-0.08,-34.5,r'$\mathrm{Gas}$',fontsize=75*FFIG)
plt.text(0.01,-34.5,r'$\mathrm{Liquid}$',fontsize=75*FFIG)
#plt.yticks(y_ticks_u_vs_x)
plt.xlim(s_lim)
plt.ylim(y_lim_)
plt.xlabel(label_s)
plt.ylabel(label_u)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_data_outside_injector_z_upper.pdf')
plt.show()
plt.close()
