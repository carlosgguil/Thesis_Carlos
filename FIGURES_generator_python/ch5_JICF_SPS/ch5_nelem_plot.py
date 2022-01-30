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
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['text.usetex'] = True

figsize_ = (FFIG*26,FFIG*16)
#figsize_ = (FFIG*20,FFIG*13)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/JICF_nelem_evolution/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/nelem evolution/'



x_label_ = r'$t^{\prime}$'
#y_label_ = '$\# ~\mathrm{elements} ~(10^6$)'
y_label_ = '$N_\mathrm{elements} ~(10^6$)'


#tau_ph_UG100 = 0.019
#tau_ph_UG75  = 0.026

tau_dr_UG75_DX10  = 0.2952
tau_dr_UG75_DX20  = 0.3558 #0.4567
tau_dr_UG100_DX10 = 0.2187
tau_dr_UG100_DX20 = 0.2584 #0.3628
tau_dr_UG100_DX10_NO_TURB = 0.2187
tau_dr_UG100_DX20_NO_TURB = 0.2602 #(X - 8.5041243834479782E-003)*1e3


'''
tau_dr_UG75_DX10  = 1
tau_dr_UG75_DX20  = 1
tau_dr_UG100_DX10 = 1
tau_dr_UG100_DX20 = 1
'''

#%% Read files

time_all_cases  = []
nelem_all_cases = []

# UG100_DX20
df = pd.read_csv(folder + 'nelem_UG100_dx20.csv')
time_UG100_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX20
nelem_UG100_DX20 = df['nelem'].values/1e6

# UG100_DX20_no_turb (CHECK IT)
df = pd.read_csv(folder + 'nelem_UG100_dx20_no_turb.csv')
time_UG100_DX20_no_turb  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX20_NO_TURB
nelem_UG100_DX20_no_turb = df['nelem'].values/1e6

# UG100_DX10
df = pd.read_csv(folder + 'nelem_UG100_dx10.csv')
time_UG100_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX10
nelem_UG100_DX10 = df['nelem'].values/1e6



# UG100_DX10_no_turb (CHECK IT)
df = pd.read_csv(folder + 'nelem_UG100_dx10_no_turb.csv')
time_UG100_DX10_no_turb  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG100_DX10_NO_TURB
nelem_UG100_DX10_no_turb = df['nelem'].values/1e6

# UG75_DX20
df = pd.read_csv(folder + 'nelem_UG75_dx20.csv')
time_UG75_DX20  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG75_DX20
nelem_UG75_DX20 = df['nelem'].values/1e6

# UG100_DX10
df = pd.read_csv(folder + 'nelem_UG75_dx10.csv')
time_UG75_DX10  = (df['total_time'].values - df.iloc[0]['total_time'])*1e3/tau_dr_UG75_DX10
nelem_UG75_DX10 = df['nelem'].values/1e6
# OJO: filtro para coger hasta solucion 26 (t' < 3.6431)
tp_limit = 3.6431
t_tmp = []; nel_tmp = []
i = 0; FOUND_TP_LIMIT = False
while not FOUND_TP_LIMIT:
    t_i = time_UG75_DX10[i]
    if t_i <= tp_limit:
        t_tmp.append(t_i)
        nel_tmp.append(nelem_UG75_DX10[i])
    else:
        FOUND_TP_LIMIT = True
    i += 1
time_UG75_DX10 = t_tmp
nelem_UG75_DX10 = nel_tmp

#%% Obtain values to get slopes of linear region

tp_cases  = [time_UG100_DX20 , time_UG100_DX10 , time_UG75_DX20 , time_UG75_DX10 ]
nel_cases = [nelem_UG100_DX20, nelem_UG100_DX10, nelem_UG75_DX20, nelem_UG75_DX10]
tp_lower = 0.6
tp_upper = 0.7
dt = tp_upper - tp_lower

nel_lower = []
nel_upper = []
slopes_linear = []
slopes_semilog = []
intercepts_semilog = []
for k in range(len(tp_cases)):
    
    tp_cases_current = tp_cases[k]
    nel_cases_current  = nel_cases[k]

    found_lower = False
    found_upper = False
    i = 0
    while (not found_lower):
        tp_i = tp_cases_current[i]
        if tp_i >= tp_lower:
            found_lower = True
            nel_lower_case = nel_cases_current[i]
        i += 1
    nel_lower.append(nel_lower_case)
    
    i = 0
    while (not found_upper):
        tp_i = tp_cases_current[i]
        if tp_i > tp_upper:
            found_upper = True
            nel_upper_case = nel_cases_current[i-1]
        i += 1
    nel_upper.append(nel_upper_case)
    
    m_linear = (nel_upper_case - nel_lower_case)/dt
    m_semilog = (np.log(nel_upper_case) - np.log(nel_lower_case))/dt
    n_semilog = nel_upper_case*np.exp(-1*m_semilog*tp_upper)
    
    slopes_linear.append(m_linear)
    slopes_semilog.append(m_semilog)
    intercepts_semilog.append(n_semilog)



#%% Fix UG75_DX20

# Define time steps to add
dt_simus = 1.5E-007
dN_iters_logs = 5
factor = 50
dt_to_place = dt_simus*dN_iters_logs*1e3/tau_dr_UG75_DX20*factor

# define v_mean and v_rms
index = 2000
v_mean = np.mean(nelem_UG75_DX20[index:])
v_RMS  = np.std(nelem_UG75_DX20[index:])*0.25

# Find times where to locate more points
time_UG75_DX20_fixed = [time_UG75_DX20[0]]
nelem_UG75_DX20_fixed = [nelem_UG75_DX20[0]]
for i in range(1,len(time_UG75_DX20)):
    t_i = time_UG75_DX20[i]
    nel_i = nelem_UG75_DX20[i]
    dt_i = t_i - time_UG75_DX20[i-1]
    if dt_i > 1: # point of log_37
        t_j = time_UG75_DX20[i-1]+dt_to_place
        REACHED_LOG_37 = False
        while (not REACHED_LOG_37):
            r   = np.random.normal()
            nel_j = v_mean + r*v_RMS
            
            time_UG75_DX20_fixed.append(t_j)
            nelem_UG75_DX20_fixed.append(nel_j)
            t_jm1 = t_j
            t_j = t_jm1 + dt_to_place
            if t_j >= t_i:
                REACHED_LOG_37 = True
                
    time_UG75_DX20_fixed.append(t_i)    
    nelem_UG75_DX20_fixed.append(nel_i)
    
    
# Plot figure
plt.figure(figsize=figsize_)
#plt.plot([1]*2,[0,1e4],'--k')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, 'b', label='Usual')
plt.plot(time_UG75_DX20_fixed,nelem_UG75_DX20_fixed, 'r', label='Fixed')
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


#%% 
plt.rcParams['ytick.minor.visible'] = True

# Full figure
plt.figure(figsize=figsize_)
plt.plot([1]*2,[0,1e4],'--k')

plt.plot(time_UG75_DX10,nelem_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20_fixed,nelem_UG75_DX20_fixed, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG100_DX20,nelem_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX20_no_turb,nelem_UG100_DX20_no_turb, '--k',label='$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$')
#plt.plot(time_UG100_DX10_no_turb,nelem_UG100_DX10_no_turb, '--b',label='$\mathrm{UG}100\_\mathrm{DX}10\_NOT$')

plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.xticks([0,5,10,15,20])
plt.ylabel(y_label_)
#plt.xlim(1e-1,11)
plt.ylim(nelem_UG75_DX10[0],3e3)
#plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
#plt.savefig(folder_manuscript + 'JICF_nelem_increase.eps',format='eps',dpi=1000)
plt.savefig(folder_manuscript + 'JICF_nelem_increase.pdf',format='pdf')
plt.show()
plt.close()





# Zoom in t' \in [0, 2]
plt.figure(figsize=figsize_)
ax = plt.gca()
'''
ax.add_patch(Rectangle((0.005, 65), 0.490, 50, fill=None, alpha=1, 
                       linewidth=5*FFIG,color='black',zorder=1e6))
ax.add_patch(Rectangle((0.501, 80), 0.2, 1.5e2, fill=None, alpha=1, 
                       linewidth=5*FFIG,color='blue',zorder=1e6))
'''
plt.plot([1]*2,[0,1e4],'--k')
plt.plot(time_UG75_DX10,nelem_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG100_DX20,nelem_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX20_no_turb,nelem_UG100_DX20_no_turb, '--k',label='$\mathrm{UG}100\_\mathrm{DX}20\_\mathrm{NT}$')
#Rectangle((0.0,100),0.25,1e3,fill='k',alpha=1)
plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.ylabel(y_label_)
plt.xlim(0,2)
plt.ylim(nelem_UG75_DX10[0],2e3)
plt.yscale('log')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_nelem_increase_t_in_0_2.pdf',format='pdf')
plt.show()
plt.close()


# Zoom in t' \in [0, 0.5]
plt.figure(figsize=figsize_)
plt.plot(time_UG100_DX20,nelem_UG100_DX20, 'r', label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, 'b', label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, 'g', label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG75_DX10,nelem_UG75_DX10, 'y', label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.xlabel(x_label_)
#plt.xlabel("$t$")
plt.ylabel(y_label_)
plt.xlim(0,0.5)
plt.ylim(nelem_UG75_DX10[0],100)
#plt.yscale('log')
#plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
#plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_nelem_increase_t_in_0_0p5.pdf',format='pdf')
plt.show()
plt.close()

# Zoom in t' \in [0, 0.3]
plt.figure(figsize=figsize_)
plt.plot(time_UG100_DX20,nelem_UG100_DX20, label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG75_DX10,nelem_UG75_DX10, label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.xlabel(x_label_)
plt.ylabel(y_label_)
plt.xlim(0,0.3)
plt.ylim(nelem_UG75_DX10[0],72)
#plt.yscale('log')
#plt.legend(loc='best')
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_nelem_increase_t_in_0_0p3.pdf',format='pdf')
plt.show()
plt.close()


#%% Zoom in linearly (logarithmically) evolving region

#% Obtain slope lines
# Intercept for slope lines
m_dx10 = 3.15 ; n_dx10 = 22
#m_dx20 = 1.6  ; n_dx20 = 39
m_dx20 = 0.9  ; n_dx20 = 48

# Time values for slope lines
tp_dx10 = np.array([0.555,0.595])
tp_dx20 = np.array([0.605,0.645])

# Slope lines
nel_dx10 = n_dx10*np.exp(m_dx10*tp_dx10)
nel_dx20 = n_dx20*np.exp(m_dx20*tp_dx20)

plt.figure(figsize=figsize_)
ax = plt.gca()
t = ax.get_xaxis().get_major_ticks()
for tick in ax.get_xaxis().get_major_ticks():
    tick.set_pad(20.)
plt.plot(tp_dx10, nel_dx10, '-.k')
plt.plot(tp_dx20, nel_dx20, '-.k')
#plt.plot(tp_dx20, nel_dx20, '-.k')
plt.plot(time_UG100_DX20,nelem_UG100_DX20, label='$\mathrm{UG}100\_\mathrm{DX}20$')
plt.plot(time_UG100_DX10,nelem_UG100_DX10, label='$\mathrm{UG}100\_\mathrm{DX}10$')
plt.plot(time_UG75_DX20,nelem_UG75_DX20, label='$\mathrm{UG}75\_\mathrm{DX}20$')
plt.plot(time_UG75_DX10,nelem_UG75_DX10, label='$\mathrm{UG}75\_\mathrm{DX}10$')
plt.text(0.56,1.45e2,r'$m_{\Delta x10}$',fontsize=70*FFIG)
#plt.text(0.61,1.12e2,r'$m_{\Delta x20}$',fontsize=70*FFIG)
plt.text(0.61,0.9e2,r'$m_{\Delta x20}$',fontsize=70*FFIG)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
plt.xlim(0.5,0.7)
#plt.ylim(80,2e2)
plt.ylim(70,2e2)
#plt.xlim(0.5,1)
#plt.ylim(80,4e2)
plt.yscale('log')
ax.yaxis.set_minor_locator(tck.FixedLocator([8e1,2e2]))
plt.grid(which='major',linestyle='-',linewidth=4*FFIG)
plt.grid(which='minor',linestyle='--')
plt.tight_layout()
plt.savefig(folder_manuscript + 'JICF_nelem_increase_t_in_0p5_1.pdf',format='pdf')
plt.show()
plt.close()

