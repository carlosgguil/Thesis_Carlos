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

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/volume_evolution/'



x_label_ = r'$t^{\prime}$'
#y_label_ = '$\# ~\mathrm{elements} ~(10^6$)'
y_label_ = r'$V_l ~[\mathrm{mm}^3$]'



tau_dr_DX15  = 1
tau_dr_DX10  = 1
tau_dr_DX07p5 = 1



#%% Read files

time_all_cases  = []
liquid_volume_all_cases = []

# DX15
df = pd.read_csv(folder + 'liquid_volume_BIMER_DX15.csv')
time_DX15  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX15
liquid_volume_DX15 = df['volume'].values*1e9

# DX10
df = pd.read_csv(folder + 'liquid_volume_BIMER_DX10.csv')
time_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX10
liquid_volume_DX10 = df['volume'].values*1e9

# DX07p5
df = pd.read_csv(folder + 'liquid_volume_BIMER_DX07p5.csv')
time_DX07p5  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX07p5
liquid_volume_DX07p5 = df['volume'].values*1e9



#%% 
plt.rcParams['ytick.minor.visible'] = True

# Full figure
plt.figure(figsize=figsize_)
#plt.plot([1]*2,[0,1e4],'--k')

plt.plot(time_DX15, liquid_volume_DX15, 'y', label='$\mathrm{DX}15$')
plt.plot(time_DX10, liquid_volume_DX10, 'g', label='$\mathrm{DX}10$')
plt.plot(time_DX07p5, liquid_volume_DX07p5, 'b', label='$\mathrm{DX}07$')

plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.xticks([0,0.5,1,1.5])
plt.ylabel(y_label_)
#plt.xlim(-1E-2,2)
#plt.ylim(nelem_DX15[0],1e3)
plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
#plt.savefig(folder_manuscript + 'JICF_nelem_increase.eps',format='eps',dpi=1000)
plt.savefig(folder_manuscript + 'BIMER_liquid_volume_increase.pdf',format='pdf')
plt.show()
plt.close()


