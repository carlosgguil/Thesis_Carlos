
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

import sys
sys.path.append('C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/LGS_previous_numerical_works/')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from function_numerical_data_past_works import read_cases_integrated_profiles

# Change size of figures 
FFIG = 0.5

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 80*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 20*FFIG
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#rc('text.latex', preamble='\usepackage{color}')


figsize_flux_y = (FFIG*25,FFIG*15)
figsize_SMD_y  = (FFIG*25,FFIG*15)
figsize_flux_z = (FFIG*18,FFIG*21)
figsize_SMD_z  = (FFIG*18,FFIG*21)




folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/previous_numerical_results/'
folder_numerics = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/LGS_previous_numerical_works/'
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'






#%% Define cases, plot parameters


label_y =  r'$y~[\mathrm{mm}]$'
label_z =  r'$z~[\mathrm{mm}]$'
#label_SMD =  r'$SMD~[\mu \mathrm{m}]$'
label_SMD = r'$\langle SMD \rangle ~[\mu \mathrm{m}]$'
label_ql = r'$q_l~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'
label_ql =  r'$\langle q_l \rangle ~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'

lims_ql = (-0.05,3)
lims_SMD_z = (0,60)
lims_SMD_y = (0,50)
lims_z = (0,25)
lims_y = (-12.5,12.5)


formats = ['k','b','r','y']
labels_cases = [r'$\mathrm{Rachner}~2002$', r'$\mathrm{Jaegle}~2009$', 
                r'$\mathrm{Senoner}~2010$', r'$\mathrm{Eckel}~2016$']

labels_expe = r'$\mathrm{Expe}$'
format_expe = 'ks'

width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG


#%% Read numerical data

y_q, q_y, y_SMD, SMD_y, z_q, q_z, z_SMD, SMD_z = read_cases_integrated_profiles(folder_numerics)

#%% Read experimental data

df_y_expe = pd.read_csv(folder_expe+'1210_01_data_integrated_z_exp.csv')
df_z_expe = pd.read_csv(folder_expe+'1210_01_data_integrated_y_exp.csv')
    
y_expe = df_y_expe['y_values'].values
q_y_expe = df_y_expe['flux_y_exp'].values
SMD_y_expe = df_y_expe['SMD_y_exp'].values
    
z_expe = df_z_expe['z_values'].values
q_z_expe = df_z_expe['flux_z_exp'].values
SMD_z_expe = df_z_expe['SMD_z_exp'].values
    

# estimate errors
error_SMD  = 0.14
error_flux = 0.2

error_q_y_expe = q_y_expe*error_flux
error_q_z_expe = q_z_expe*error_flux
error_SMD_y_expe = SMD_y_expe*error_SMD
error_SMD_z_expe = SMD_z_expe*error_SMD

#%% Plot flux along z

plt.figure(figsize=figsize_flux_z)
plt.plot(q_z_expe, z_expe,format_expe,label=labels_expe)
plt.errorbar(q_z_expe, z_expe, xerr=error_q_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(z_q)):
    plt.plot(q_z[i], z_q[i], formats[i], label=labels_cases[i])
plt.legend(loc='best')
plt.xlabel(label_ql)
plt.xlim(lims_ql)
plt.ylabel(label_z)
plt.ylim(lims_z)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'flux_profiles_along_z.pdf')
plt.show()
plt.close()



#%% Plot SMD along z

plt.figure(figsize=figsize_SMD_z)
plt.plot(SMD_z_expe, z_expe,format_expe,label=labels_expe)
plt.errorbar(SMD_z_expe, z_expe, xerr=error_SMD_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(z_SMD)):
    if z_SMD[i] is not None:
        plt.plot(SMD_z[i], z_SMD[i], formats[i], label=labels_cases[i])
#plt.legend(loc='best')
plt.xlabel(label_SMD)
plt.xlim(lims_SMD_z)
plt.ylabel(label_z)
plt.ylim(lims_z)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_profiles_along_z.pdf')
plt.show()
plt.close()

'''
plt.figure(figsize=figsize_SMD_y)
plt.plot(z_expe, SMD_z_expe, format_expe,label=labels_expe)
for i in range(len(z_SMD)):
    if z_SMD[i] is not None:
        plt.plot(z_SMD[i], SMD_z[i], formats[i], label=labels_cases[i])
plt.legend(loc='best')
plt.ylabel(label_SMD)
plt.ylim(lims_SMD_z)
plt.xlabel(label_z)
plt.xlim(lims_z)
plt.grid()
plt.show()
plt.close()
'''


#%% Plot fluxes along y

plt.figure(figsize=figsize_flux_y)
plt.plot(y_expe, q_y_expe,format_expe,label=labels_expe)
plt.errorbar(y_expe, q_y_expe, yerr=error_q_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(y_q)):
    plt.plot(y_q[i], q_y[i], formats[i], label=labels_cases[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.xlim(lims_y)
plt.ylabel(label_ql)
plt.ylim(lims_ql)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'flux_profiles_along_y.pdf')
plt.show()
plt.close()

#%% Plot SMD along y


plt.figure(figsize=figsize_SMD_y)
plt.plot(y_expe, SMD_y_expe,format_expe,label=labels_expe)
plt.errorbar(y_expe, SMD_y_expe, yerr=error_SMD_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(y_SMD)):
    if y_SMD[i] is not None:
        plt.plot(y_SMD[i], SMD_y[i], formats[i], label=labels_cases[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.xlim(lims_y)
plt.ylabel(label_SMD)
plt.ylim(lims_SMD_y)
plt.yticks(np.linspace(0,50,6))
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_profiles_along_y.pdf')
plt.show()
plt.close()
