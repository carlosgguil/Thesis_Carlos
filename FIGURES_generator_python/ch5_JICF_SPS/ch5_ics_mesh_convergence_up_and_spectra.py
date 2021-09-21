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

T = 1E-6
#%% Cases


# Main folders
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_ics_mesh_convergence_probes/'
folder = 'C:/Users/d601630/Desktop/Ongoing/ICS_study/frequential_analyses/cases_probes/'

# Cases
case_DX1p0 = folder + 'mesh_refined_DX1p0_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5 = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_DX0p5_no_turb = folder + 'mesh_refined_DX0p5_ics_no_actuator_flat_BL_no_turbulence/'
case_DX0p3 = folder + 'mesh_refined_DX0p3_ics_no_actuator_flat_BL_with_turbulence_L3p00_up2p7/'
case_OP2   = folder + '2nd_op_mesh_DX0p5/'
         
cases = [case_DX1p0, case_DX0p5, case_DX0p5_no_turb, case_DX0p3, case_OP2]


# Labels
labels_ = [r'$\mathrm{Probe}~\mathrm{A}$', r'$\mathrm{Probe}~\mathrm{B}$']
    

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
tau_ft_op2 = 1.6

# x labels for up
t_threshold = 8
t_lim_min   = t_threshold
t_lim_max   = t_threshold + 1

# ylabels for up
y_ticks_up = [-4.0,-2.0,0.0,2.0,4.0]
y_lim_up   = [-5.0,5.0]

#%% Read probes
time_all_cases_line_0 = []
u_all_cases_line_0 = []
xf_all_cases_line_0 = []
y_FFT_all_cases_line_0 = []


time_all_cases_line_inj = []
u_all_cases_line_inj = []
xf_all_cases_line_inj = []
y_FFT_all_cases_line_inj = []

for i in range(len(cases)):
    case = cases[i]
    
    df_line_0_up        = pd.read_csv(case+'data_line_0_up.csv')
    df_line_0_spectra   = pd.read_csv(case+'data_line_0_spectra.csv')
    df_line_inj_up      = pd.read_csv(case+'data_line_inj_up.csv')
    df_line_inj_spectra = pd.read_csv(case+'data_line_inj_spectra.csv')

    # Probe at channel inlet (line_0)
    time_line_0 = (df_line_0_up['t'].values - df_line_0_up['t'][0])*1e3/tau_ft + t_threshold
    time_all_cases_line_0.append(time_line_0)
    u_all_cases_line_0.append(df_line_0_up['up'].values)
    xf_all_cases_line_0.append(df_line_0_spectra['xf'].values)
    y_FFT_all_cases_line_0.append(df_line_0_spectra['y_FFT'].values)

    # Probe at injector lip (line_inj)
    time_line_inj = (df_line_inj_up['t'].values - df_line_inj_up['t'].values[0])*1e3/tau_ft + t_threshold
    time_all_cases_line_inj.append(time_line_inj)
    u_all_cases_line_inj.append(df_line_inj_up['up'].values)
    xf_all_cases_line_inj.append(df_line_inj_spectra['xf'].values)
    y_FFT_all_cases_line_inj.append(df_line_inj_spectra['y_FFT'].values)

    


#%% Plot u' signal
    


# dx = 1.0 mm
j = 0
plt.figure(figsize=figsize_)
plt.plot(time_all_cases_line_0[j]  ,u_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(time_all_cases_line_inj[j],u_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
plt.xlim(t_lim_min,t_lim_max)
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx1p0.pdf')
plt.savefig(folder_manuscript+'up_dx1p0.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% 
t = time_all_cases_line_0[j]*tau_ft/1e3
u = u_all_cases_line_0[j]

up = u - np.mean(u)
N = len(t)
xf = fftfreq(N, T)[:N//2]
yf = fft(up)
yplot = np.abs(yf[0:N //2])




# Plot FFT transform (linear scale)
plt.figure(figsize=figsize_)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.plot(xf/1000, yplot)
plt.xlim(0,100000/1000)
plt.xlabel('f [kHz]')
plt.ylabel(r"FFT ($u'$)")
plt.legend(loc='best')
plt.grid()
plt.show()
plt.close()

# Obtain energy by integrating u'
E_u = 0
for i in range(len(t)-1):
    dt = t[i+1] - t[i]
    up2_t   = up[i]**2
    up2_tp1 = up[i+1]**2
    E_u += 0.5*(up2_t + up2_tp1)*dt
    
E_u = E_u/(t[-1]-t[0])

# obtain energy by integrating spectrum
E_f = 0
for i in range(len(yplot)-1):
    df = xf[i+1]  - xf[i]
    yf = yplot[i]
    yfp1 = yplot[i+1]
    E_f += 0.5*(yf+yfp1)*df

#%%

# dx = 0.5 mm
j = 1
plt.figure(figsize=figsize_)
plt.plot(time_all_cases_line_0[j]  ,u_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(time_all_cases_line_inj[j],u_all_cases_line_inj[j], c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlim(t_lim_min,t_lim_max)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx0p5.pdf')
plt.savefig(folder_manuscript+'up_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# dx = 0.5 mm no turbulence
j = 2
plt.figure(figsize=figsize_)
plt.plot(time_all_cases_line_0[j]  ,u_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(time_all_cases_line_inj[j],u_all_cases_line_inj[j], c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlim(t_lim_min,t_lim_max)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx0p5_no_turb.pdf')
plt.savefig(folder_manuscript+'up_dx0p5_no_turb.eps',format='eps',dpi=1000)
plt.show()
plt.close()


# dx = 0.3 mm
j = 3
plt.figure(figsize=figsize_)
plt.plot(time_all_cases_line_0[j]  ,u_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(time_all_cases_line_inj[j],u_all_cases_line_inj[j], c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlim(t_lim_min,t_lim_max)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_dx0p3.pdf')
plt.savefig(folder_manuscript+'up_dx0p3.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# OP2
j = 4
plt.figure(figsize=figsize_)
plt.plot(time_all_cases_line_0[j]  ,u_all_cases_line_0[j],   c1, label=labels_[0])
plt.plot(time_all_cases_line_inj[j],u_all_cases_line_inj[j], c2, label=labels_[1])
#plt.ylim(0, 10)
plt.xlim(t_lim_min,t_lim_max)
plt.xlabel(x_label_up)
plt.ylabel(y_label_up)
plt.ylim(y_lim_up)
plt.yticks(y_ticks_up)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'up_OP2.pdf')
plt.savefig(folder_manuscript+'up_OP2.eps',format='eps',dpi=1000)
plt.show()
plt.close()



#%% Plot FFTs


# dx=1.5
j = 0

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,40000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx1p0.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx1p0.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
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
plt.plot(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,40000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel('f [kHz]')
plt.ylabel(r"FFT ($u'$)")
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5.eps',format='eps',dpi=1000)
plt.show()
plt.close()



# dx=0.5 no turbulence
j = 2

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,40000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5_no_turb.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p5_no_turb.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5_no_turb.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_dx0p5_no_turb.eps',format='eps',dpi=1000)
plt.show()
plt.close()




# dx=0.3
j = 3

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,40000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p3.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_dx0p3.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
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


# OP2
j = 4

# linear scale
plt.figure(figsize=figsize_)
plt.plot(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.plot(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(0,40000/1000)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_linear_scale_OP2.pdf')
plt.savefig(folder_manuscript+'spectra_linear_scale_OP2.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# log scale
plt.figure(figsize=figsize_)
plt.loglog(xf_all_cases_line_0[j]/1000  , y_FFT_all_cases_line_0[j]  , c1, label=labels_[0])
plt.loglog(xf_all_cases_line_inj[j]/1000, y_FFT_all_cases_line_inj[j], c2, label=labels_[1])
plt.xlim(1e-1,1E3)
plt.ylim(1e-5,1e1)
plt.xlabel(x_label_FFT)
plt.ylabel(y_label_FFT)
#plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'spectra_log_scale_OP2.pdf')
plt.savefig(folder_manuscript+'spectra_log_scale_OP2.eps',format='eps',dpi=1000)
plt.show()
plt.close()