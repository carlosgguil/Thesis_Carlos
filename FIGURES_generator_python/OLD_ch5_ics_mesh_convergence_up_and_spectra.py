# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:08:21 2021

@author: d601630
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.fftpack import fftfreq

FFIG = 0.5
# rcParams for plots
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

figsize_ = (FFIG*20,FFIG*13)

#%% Cases
T = 1.0E-6

# Select Z point [mm]
Z = 8.5

# Main folders
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_probes/'
folder = 'C:/Users/d601630/Desktop/Ongoing/ICS_study/frequential_analyses/cases_probes/'

# Cases
case_DX1p0 = folder + 'irene_mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5 = folder + 'irene_mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p3 = folder + 'irene_mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
         
cases = [case_DX1p0, case_DX0p5, case_DX0p3]


# Labels
labels_ = ['Channel inlet', 'Injector lip']
    

# axis labels
x_label_up  = r"$t/\tau_\mathrm{fl}$"
y_label_up  = "$u' ~ [m . s^{-1}]$"
x_label_FFT = "$f ~[kHz]$"
y_label_FFT = "$\mathrm{FFT} (u')$"

# Format for lines
c1 = 'k'
c2 = '--b'

# flow-through time [ms]
tau_ft = 1.2

# ylabels for
y_ticks_up = [-5.0,-2.5,0.0,2.5,5.0]
y_lim_up   = [-6.0,6.0]

#%% Read probes
time_all_cases_line_0 = []
u_all_cases_line_0 = []
time_all_cases_line_2 = []
u_all_cases_line_2 = []
time_all_cases_line_inj = []
u_all_cases_line_inj = []

for i in range(len(cases)):
    case = cases[i]

    # Probe at channel inlet (line_0)
    probe = pd.read_csv(case+'line_0_U.dat',sep='(?<!\\#)\s+',engine='python')      
    probe.columns =  [name.split(':')[1] for name in probe.columns]
    time_all = probe['total_time'].values*1e3/tau_ft
    u_all    = probe['U(1)'].values
        
    time = []
    u    = []
    for i in range(len(probe)):
        z_i = probe.loc[i]['Z']*1e3
        if round(z_i,2) == round(Z,2):
            time.append(time_all[i])
            u.append(u_all[i])
    time -= time[0]
    
    time_all_cases_line_0.append(time)
    u_all_cases_line_0.append(u)
    
    # Probe at channel middle (line_2)   
    probe = pd.read_csv(case+'line_2_U.dat',sep='(?<!\\#)\s+',engine='python')      
    probe.columns =  [name.split(':')[1] for name in probe.columns]
    time_all = probe['total_time'].values*1e3/tau_ft
    u_all    = probe['U(1)'].values
        
    time = []
    u    = []
    for i in range(len(probe)):
        z_i = probe.loc[i]['Z']*1e3
        if round(z_i,2) == round(Z,2):
            time.append(time_all[i])
            u.append(u_all[i])
    time -= time[0]
    
    time_all_cases_line_2.append(time)
    u_all_cases_line_2.append(u)
        
    
    # Probe at nozzle lip (line_inj)
    probe = pd.read_csv(case+'line_inj_U.dat',sep='(?<!\\#)\s+',engine='python')      
    probe.columns =  [name.split(':')[1] for name in probe.columns]
    time_all = probe['total_time'].values*1e3/tau_ft
    u_all    = probe['U(1)'].values
        
    time = []
    u    = []
    for i in range(len(probe)):
        z_i = probe.loc[i]['Z']*1e3
        if round(z_i,2) == round(Z,2):
            time.append(time_all[i])
            u.append(u_all[i])
    time -= time[0]
    
    time_all_cases_line_inj.append(time)
    u_all_cases_line_inj.append(u)
        
    

#%% Filter repeated time values
p_time_all_cases_line_0 = []
p_u_all_cases_line_0 = []
p_time_all_cases_line_2 = []
p_u_all_cases_line_2 = []
p_time_all_cases_line_inj = []
p_u_all_cases_line_inj = []

for i in range(len(cases)):
    
    # line 0
    time = time_all_cases_line_0[i]
    u    = u_all_cases_line_0[i]
    
    p_time  = []; time_max = -1
    p_u     = []
    for j in range(len(time)):
        t = time[j]
        if t > time_max:
            p_time.append(t)
            p_u.append(u[j])
            time_max = t

    p_time_all_cases_line_0.append(np.array(p_time))
    p_u_all_cases_line_0.append(np.array(p_u))
    
    
    # line 2
    time = time_all_cases_line_2[i]
    u    = u_all_cases_line_2[i]
    
    p_time  = []; time_max = -1
    p_u     = []
    for j in range(len(time)):
        t = time[j]
        if t > time_max:
            p_time.append(t)
            p_u.append(u[j])
            time_max = t

    p_time_all_cases_line_2.append(np.array(p_time))
    p_u_all_cases_line_2.append(np.array(p_u))
    
    
    # line inj
    time = time_all_cases_line_inj[i]
    u    = u_all_cases_line_inj[i]
    
    p_time  = []; time_max = -1
    p_u     = []
    for j in range(len(time)):
        t = time[j]
        if t > time_max:
            p_time.append(t)
            p_u.append(u[j])
            time_max = t

    p_time_all_cases_line_inj.append(np.array(p_time))
    p_u_all_cases_line_inj.append(np.array(p_u))
    
# Substract mean component
for i in range(len(cases)):
    p_u_all_cases_line_0[i]   -= np.mean(p_u_all_cases_line_0[i])
    p_u_all_cases_line_2[i]   -= np.mean(p_u_all_cases_line_2[i])
    p_u_all_cases_line_inj[i] -= np.mean(p_u_all_cases_line_inj[i])

#%% Plot u' signal
    


# dx = 1.0 mm
j = 0
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases_line_0[j]  ,p_u_all_cases_line_0[j],  c1, label=labels_[0])
plt.plot(p_time_all_cases_line_inj[j],p_u_all_cases_line_inj[j],c2, label=labels_[1])
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
#plt.xlim([0,2])
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx1p0.pdf')
plt.savefig(folder_manuscript+'up_dx1p0.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# dx = 0.5 mm
j = 1
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases_line_0[j]  ,p_u_all_cases_line_0[j],  c1, label=labels_[0])
plt.plot(p_time_all_cases_line_inj[j],p_u_all_cases_line_inj[j],c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
#plt.xlim([0,2])
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx0p5.pdf')
plt.savefig(folder_manuscript+'up_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# dx = 0.3 mm
j = 2
plt.figure(figsize=figsize_)
plt.plot(p_time_all_cases_line_0[j]  ,p_u_all_cases_line_0[j],  c1, label=labels_[0])
plt.plot(p_time_all_cases_line_inj[j],p_u_all_cases_line_inj[j],c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
#plt.xlim([0,2])
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx0p3.pdf')
plt.savefig(folder_manuscript+'up_dx0p3.eps',format='eps',dpi=1000)
plt.show()
plt.close()


#%% Get FFTs

xf_all_cases_line_0 = []
yplot_all_cases_line_0 = []
xf_all_cases_line_2 = []
yplot_all_cases_line_2 = []
xf_all_cases_line_inj = []
yplot_all_cases_line_inj = []
for i in range(len(cases)):
    
    # line 0
    p_time = p_time_all_cases_line_0[i]
    p_u    = p_u_all_cases_line_0[i]
    
    p_u = p_u - np.mean(p_u)
    N = len(p_time)
    
    yf = fft(p_u)
    xf = fftfreq(N, T)[:N//2]
    yplot = np.abs(yf[0:N //2])
    yplot = yplot/max(yplot)
    
    xf_all_cases_line_0.append(xf)
    yplot_all_cases_line_0.append(yplot)
    
    
    
    # line 2
    p_time = p_time_all_cases_line_2[i]
    p_u    = p_u_all_cases_line_2[i]
    
    p_u = p_u - np.mean(p_u)
    N = len(p_time)
    
    yf = fft(p_u)
    xf = fftfreq(N, T)[:N//2]
    yplot = np.abs(yf[0:N //2])
    yplot = yplot/max(yplot)
    
    xf_all_cases_line_2.append(xf)
    yplot_all_cases_line_2.append(yplot)
    
    
    
    # line inj
    p_time = p_time_all_cases_line_inj[i]
    p_u    = p_u_all_cases_line_inj[i]
    
    p_u = p_u - np.mean(p_u)
    N = len(p_time)
    
    yf = fft(p_u)
    xf = fftfreq(N, T)[:N//2]
    yplot = np.abs(yf[0:N //2])
    yplot = yplot/max(yplot)
    
    xf_all_cases_line_inj.append(xf)
    yplot_all_cases_line_inj.append(yplot)




#%% Plot FFTs


# dx=1.5
j = 0

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,30000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx1p0.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx1p0.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_dx1p0.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_dx1p0.eps',format='eps',dpi=1000)
plt.show()
plt.close()





# dx=0.5
j = 1

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,30000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel('f [kHz]')
plt.ylabel(r"FFT ($u'$)")
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()




# dx=0.3
j = 2

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,30000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p3.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p3.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , yplot_all_cases_line_0[j],   c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, yplot_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p3.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p3.eps',format='eps',dpi=1000)
plt.show()
plt.close()
