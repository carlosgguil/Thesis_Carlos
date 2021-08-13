# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:26:19 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*26,FFIG*16)

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/'
folder = 'C:/Users/d601630/Desktop/Ongoing/JICF/nelem evolution/'






tau_ph_UG100 = 0.019
tau_ph_UG75  = 0.026

#%% Read files

time_all_cases  = []
nelem_all_cases = []

# UG100_DX20
df = pd.read_csv(folder + 'nelem_UG100_dx20.csv')
time_UG100_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_ph_UG100
nelem_UG100_DX20 = df['nelem'].values/1e6

# UG100_DX10
df = pd.read_csv(folder + 'nelem_UG100_dx10.csv')
time_UG100_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_ph_UG100
nelem_UG100_DX10 = df['nelem'].values/1e6



# UG75_DX20
df = pd.read_csv(folder + 'nelem_UG75_dx20.csv')
time_UG75_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_ph_UG75
nelem_UG75_DX20 = df['nelem'].values/1e6

# UG100_DX10
df = pd.read_csv(folder + 'nelem_UG75_dx10.csv')
time_UG75_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_ph_UG75
nelem_UG75_DX10 = df['nelem'].values/1e6

#%% 

plt.figure(figsize=figsize_)
plt.plot(time_UG100_DX20,nelem_UG100_DX20, label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG75_DX10,nelem_UG75_DX10, label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.xlabel('$t^*$')
plt.ylabel('$\# ~\mathrm{elements} ~(10^6$)')
plt.yscale('log')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_nelem_increase.eps',format='eps',dpi=1000)
plt.show()
plt.close()


