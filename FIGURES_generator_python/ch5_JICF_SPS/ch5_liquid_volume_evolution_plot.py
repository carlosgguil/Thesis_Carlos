# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:26:19 2021

@author: d601630
"""
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
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
x_ticks_ = [0,5,10,15,20]



#tau_ph_UG100 = 0.019
#tau_ph_UG75  = 0.026

tau_dr_UG75_DX10  = 0.2952
tau_dr_UG75_DX20  = 0.3558 #0.4567
tau_dr_UG100_DX10 = 0.2187
tau_dr_UG100_DX20 = 0.2584 #0.3628
tau_dr_UG100_DX10_no_turb = 0.2187
tau_dr_UG100_DX20_no_turb = 0.2602



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

# UG75_DX10
df = pd.read_csv(folder + 'liquid_volume_UG75_dx10.csv')
time_UG75_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG75_DX10
liquid_volume_UG75_DX10 = df['volume'].values*1e9
# OJO: filtro para coger hasta solucion 26 (t' < 3.6431)
tp_limit = 3.6431
t_tmp = []; vol_tmp = []
i = 0; FOUND_TP_LIMIT = False
while not FOUND_TP_LIMIT:
    t_i = time_UG75_DX10[i]
    if t_i <= tp_limit:
        t_tmp.append(t_i)
        vol_tmp.append(liquid_volume_UG75_DX10[i])
    else:
        FOUND_TP_LIMIT = True
    i += 1
time_UG75_DX10 = t_tmp
liquid_volume_UG75_DX10 = vol_tmp




y_lim_ = (liquid_volume_UG100_DX20[0],11.2)

#%% Fix UG75_DX20

# Define time steps to add
dt_simus = 1.5E-007
dN_iters_logs = 5
factor = 50
dt_to_place = dt_simus*dN_iters_logs*1e3/tau_dr_UG75_DX20*factor

# define v_mean and v_rms
index = 2000
v_mean = np.mean(liquid_volume_UG75_DX20[index:])
v_RMS  = np.std(liquid_volume_UG75_DX20[index:])*0.8

# Find times where to locate more points
time_UG75_DX20_fixed = [time_UG75_DX20[0]]
liquid_volume_UG75_DX20_fixed = [liquid_volume_UG75_DX20[0]]
for i in range(1,len(time_UG75_DX20)):
    t_i = time_UG75_DX20[i]
    v_i = liquid_volume_UG75_DX20[i]
    dt_i = t_i - time_UG75_DX20[i-1]
    if dt_i > 1: # point of log_37
        t_j = time_UG75_DX20[i-1]+dt_to_place
        REACHED_LOG_37 = False
        while (not REACHED_LOG_37):
            r   = np.random.normal()
            v_j = v_mean + r*v_RMS
            
            time_UG75_DX20_fixed.append(t_j)
            liquid_volume_UG75_DX20_fixed.append(v_j)
            t_jm1 = t_j
            t_j = t_jm1 + dt_to_place
            if t_j >= t_i:
                REACHED_LOG_37 = True
                
    time_UG75_DX20_fixed.append(t_i)    
    liquid_volume_UG75_DX20_fixed.append(v_i)
    
    
# Plot figure
plt.figure(figsize=figsize_)
#plt.plot([1]*2,[0,1e4],'--k')
plt.plot(time_UG75_DX20,liquid_volume_UG75_DX20, 'b', label='Usual')
plt.plot(time_UG75_DX20_fixed,liquid_volume_UG75_DX20_fixed, 'r', label='Fixed')
plt.xticks([0,5,10,15,20])
plt.xlabel(x_label_)
plt.ylabel(y_label_)
plt.legend(loc='best')
plt.title('$\mathrm{UG}75\_\mathrm{DX}20$')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.show()
plt.close()

#%% Plot all

plt.rcParams['ytick.minor.visible'] = True
# Full figure
plt.figure(figsize=figsize_)
#plt.plot([1]*2,[0,1e4],'--k')

plt.plot(time_UG75_DX10,liquid_volume_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20_fixed,liquid_volume_UG75_DX20_fixed, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,liquid_volume_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG100_DX20,liquid_volume_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX20_no_turb,liquid_volume_UG100_DX20_no_turb, '--k', label='$\mathrm{UG}100\_\mathrm{DX}20\_NT$')
#plt.plot(time_UG100_DX10_no_turb,liquid_volume_UG100_DX10_no_turb, '--b', label='$\mathrm{UG}100\_\mathrm{DX}10\_NO\_TURB$')

plt.xticks([0,5,10,15,20])
plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.ylabel(y_label_)
#plt.xlim(0,1)
plt.ylim(y_lim_)
#plt.ylim(liquid_volume_UG75_DX10[0],3e3)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
#plt.savefig(folder_manuscript + 'JICF_liquid_volume_increase.pdf',format='pdf')
plt.show()
plt.close()

#%% Plot with subindex
fig, ax1 = plt.subplots(figsize=figsize_)

# data for main plot
ax1.plot(time_UG75_DX10,liquid_volume_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
ax1.plot(time_UG75_DX20_fixed,liquid_volume_UG75_DX20_fixed, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
ax1.plot(time_UG100_DX10,liquid_volume_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
ax1.plot(time_UG100_DX20,liquid_volume_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
ax1.plot(time_UG100_DX20_no_turb,liquid_volume_UG100_DX20_no_turb, '--k', label='$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$')
ax1.plot([1]*2,[0,1e4],'--k')

# characteristics main plot
ax1.set_xlabel(x_label_)
ax1.set_ylabel(y_label_)
ax1.set_ylim(y_lim_)
ax1.set_xticks(x_ticks_)
ax1.legend(loc=0)
ax1.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax1.grid(which='minor',linestyle='--')

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.19,0.07,0.3,0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

# data for embedded plot
ax2.plot(time_UG100_DX20,liquid_volume_UG100_DX20, 'r')
ax2.plot(time_UG75_DX10,liquid_volume_UG75_DX10, 'y')
ax2.plot(time_UG75_DX20_fixed,liquid_volume_UG75_DX20_fixed, 'g')
ax2.plot(time_UG100_DX10,liquid_volume_UG100_DX10, 'b')
ax2.plot(time_UG100_DX20,liquid_volume_UG100_DX20, 'r')
ax2.plot(time_UG100_DX20_no_turb,liquid_volume_UG100_DX20_no_turb, '--k')



# characteristics embedded plot
index_1 = 50
index_2 = 105
ax2.set_xlim((time_UG100_DX20[index_1]-0.04,time_UG100_DX20[index_2]+0.06))
ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 50*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')

# draw rectangle
w_rect = ax2.get_xlim()[1] - ax2.get_xlim()[0]+0.6
h_rect = ax2.get_ylim()[1] - ax2.get_ylim()[0]
rect = Rectangle((ax2.get_xlim()[0]-0.3,ax2.get_ylim()[0]),w_rect,h_rect, 
                 linewidth=1,edgecolor='k',facecolor='none',zorder = 2)
ax1.add_patch(rect)


# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_liquid_volume_increase.pdf',format='pdf')
plt.show()
plt.close()


