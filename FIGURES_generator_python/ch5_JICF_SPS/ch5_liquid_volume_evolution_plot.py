# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:26:19 2021

@author: d601630
"""
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*26,FFIG*16)
#figsize_ = (FFIG*20,FFIG*13)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/volume_evolution/'



x_label_ = r'$t^{\prime}$'
#y_label_ = '$\# ~\mathrm{elements} ~(10^6$)'
y_label_ = r'$V_l ~[\mathrm{mm}^3$]'


#tau_ph_UG100 = 0.019
#tau_ph_UG75  = 0.026

tau_dr_UG75_DX10  = 0.2952
tau_dr_UG75_DX20  = 0.3558 #0.4567
tau_dr_UG100_DX10 = 0.2187
tau_dr_UG100_DX20 = 0.2584 #0.3628
tau_dr_UG100_DX10_no_turb = 0.2187
tau_dr_UG100_DX20_no_turb = 0.2584



'''
tau_dr_UG75_DX10  = 1
tau_dr_UG75_DX20  = 1
tau_dr_UG100_DX10 = 1
tau_dr_UG100_DX20 = 1
'''

#%% Read files

time_all_cases  = []
liquid_volume_all_cases = []

# UG100_DX20
df = pd.read_csv(folder + 'liquid_volume_UG100_dx20.csv')
time_UG100_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX20
liquid_volume_UG100_DX20 = df['volume'].values*1e9

# UG100_DX20_no_turb (CHECK IT)
df = pd.read_csv(folder + 'liquid_volume_UG100_dx20_no_turb.csv')
time_UG100_DX20_no_turb  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX20_no_turb
liquid_volume_UG100_DX20_no_turb = df['volume'].values*1e9

# UG100_DX10
df = pd.read_csv(folder + 'liquid_volume_UG100_dx10.csv')
time_UG100_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX10
liquid_volume_UG100_DX10 = df['volume'].values*1e9

# UG100_DX10_no_turb (CHECK IT)
df = pd.read_csv(folder + 'liquid_volume_UG100_dx10_no_turb.csv')
time_UG100_DX10_no_turb  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX10_no_turb
liquid_volume_UG100_DX10_no_turb = df['volume'].values*1e9

# UG75_DX20
df = pd.read_csv(folder + 'liquid_volume_UG75_dx20.csv')
time_UG75_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG75_DX20
liquid_volume_UG75_DX20 = df['volume'].values*1e9

# UG100_DX10
df = pd.read_csv(folder + 'liquid_volume_UG75_dx10.csv')
time_UG75_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG75_DX10
liquid_volume_UG75_DX10 = df['volume'].values*1e9



#%% 
plt.rcParams['ytick.minor.visible'] = True

# Full figure
plt.figure(figsize=figsize_)
#plt.plot([1]*2,[0,1e4],'--k')

plt.plot(time_UG75_DX10,liquid_volume_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,liquid_volume_UG75_DX20, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,liquid_volume_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG100_DX20,liquid_volume_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX20_no_turb,liquid_volume_UG100_DX20_no_turb, '--r', label='$\mathrm{UG}100\_\mathrm{DX}20\_NT$')
#plt.plot(time_UG100_DX10_no_turb,liquid_volume_UG100_DX10_no_turb, '--b', label='$\mathrm{UG}100\_\mathrm{DX}10\_NO\_TURB$')

plt.xticks([0,5,10,15,20])
plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.ylabel(y_label_)
#plt.xlim(1e-1,11)
#plt.ylim(liquid_volume_UG75_DX10[0],3e3)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_liquid_volume_increase.pdf',format='pdf')
plt.show()
plt.close()



