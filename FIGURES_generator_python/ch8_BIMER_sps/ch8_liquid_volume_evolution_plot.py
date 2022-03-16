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

ticks_tp_label = np.linspace(0,7,8)
ticks_Vl_label = [0.6,0.65,0.7,0.75,0.8]

# Times correspond to x_c/d_inj = 6.67 #10
tau_dr_DX15  = 562e-3 #633e-3
tau_dr_DX10  = 354e-3 #428e-3
tau_dr_DX07p5 = 359e-3 #434e-3


#%% Read files

time_all_cases  = []
liquid_volume_all_cases = []

# DX15
df = pd.read_csv(folder + 'liquid_volume_BIMER_dx15p0.csv')
time_DX15  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX15
liquid_volume_DX15 = df['volume'].values*1e9

# DX10
df = pd.read_csv(folder + 'liquid_volume_BIMER_dx10p0.csv')
time_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX10
liquid_volume_DX10 = df['volume'].values*1e9

# DX07p5
df = pd.read_csv(folder + 'liquid_volume_BIMER_dx07p5.csv')
time_DX07p5  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_DX07p5
liquid_volume_DX07p5 = df['volume'].values*1e9




#%% Fix UG75_DX15
factor_tp_if_tau_changes = 633e-3/tau_dr_DX15

tp_Ql_min = 3.9*factor_tp_if_tau_changes
tp_Ql_max = 5.95*factor_tp_if_tau_changes
tp_threshold = 3.5*factor_tp_if_tau_changes

# find indexes corresponding to times
i = 0
FOUND_tp_Ql_min = False
FOUND_tp_Ql_max = False
FOUND_tp_threshold = False
while( (not FOUND_tp_Ql_min) or (not FOUND_tp_Ql_max) or (not FOUND_tp_threshold)):
    t_i = time_DX15[i]
    if (not FOUND_tp_Ql_min):
        if t_i > tp_Ql_min:
            FOUND_tp_Ql_min = True
            index_tp_Ql_min = i
    if (not FOUND_tp_Ql_max):
        if t_i > tp_Ql_max:
            FOUND_tp_Ql_max = True
            index_tp_Ql_max = i
    if (not FOUND_tp_threshold):
        if t_i > tp_threshold:
            FOUND_tp_threshold = True
            index_tp_threshold = i
    i += 1

Ql_shift_num = liquid_volume_DX15[index_tp_Ql_max] - liquid_volume_DX15[index_tp_Ql_min]
Ql_shift_den = time_DX15[index_tp_Ql_max] - time_DX15[index_tp_Ql_min]
Ql_shift     = Ql_shift_num/Ql_shift_den


# finally, shift values
t0_shift = time_DX15[index_tp_threshold]
for i in range(index_tp_threshold,len(liquid_volume_DX15)):
    Vl_shift = Ql_shift*(time_DX15[i] - t0_shift)
    V_l_actual = liquid_volume_DX15[i]
    liquid_volume_DX15[i] = V_l_actual - Vl_shift


#%% 

y_lim_Vl = (liquid_volume_DX15[0], liquid_volume_DX15[-1]+0.01)


# Full figure
plt.figure(figsize=figsize_)

plt.plot([1]*2,[0,1e4],'--k')
plt.plot(time_DX15, liquid_volume_DX15, 'k', label='$\mathrm{DX}15$')
plt.plot(time_DX10, liquid_volume_DX10, 'b', label='$\mathrm{DX}10$')
#plt.plot(time_DX07p5, liquid_volume_DX07p5, 'b', label='$\mathrm{DX}07$')

plt.xlabel(x_label_)
plt.xlim((0,7))
plt.xticks(ticks_tp_label)
plt.ylabel(y_label_)
plt.ylim(y_lim_Vl)
plt.yticks(ticks_Vl_label)
#plt.ylim(nelem_DX15[0],1e3)
plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
#plt.grid(which='minor',linestyle='--')
plt.tight_layout()
#plt.savefig(folder_manuscript + 'JICF_nelem_increase.eps',format='eps',dpi=1000)
plt.savefig(folder_manuscript + 'BIMER_liquid_volume_increase.pdf',format='pdf')
plt.show()
plt.close()


