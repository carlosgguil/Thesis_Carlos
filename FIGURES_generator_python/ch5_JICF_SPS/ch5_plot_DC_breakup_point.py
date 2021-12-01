# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:22:20 2021

@author: d601630
"""


#from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.fftpack import fftfreq


FFIG = 0.5
'''
plt.rcParams['xtick.labelsize'] = 80*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 60*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG
plt.rcParams['legend.loc']      = 'best'
plt.rcParams['text.usetex'] = True
plt.rcParams['legend.framealpha'] = 1.0
'''

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]
#rc('text.latex', preamble='\usepackage{color}')

d_inj = 0.45
T     = 1.5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_dense_core_modeling/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/DC characterization/data_breakup_points/'


cases = [folder+'breakup_point_uG75_dx10m.csv',
         folder+'breakup_point_uG75_dx20m.csv',
         folder+'breakup_point_uG100_dx10m.csv',
         folder+'breakup_point_uG100_dx20m.csv',
         folder+'breakup_point_uG100_dx20m_no_turb.csv']

labels_ = [r'$\mathrm{UG75}\_\mathrm{DX10}$' ,r'$\mathrm{UG75}\_\mathrm{DX20}$',
           r'$\mathrm{UG100}\_\mathrm{DX10}$' , r'$\mathrm{UG100}\_\mathrm{DX20}$',
           r'$\mathrm{UG100}\_\mathrm{DX20}\_\mathrm{NO}\_\mathrm{TURB}$']

   
save_labels = ['UG75_DX10',  'UG75_DX20',
               'UG100_DX10', 'UG100_DX20','UG100_DX20_NO_TURB']

# Characteristic times to non-dimensionalize
tau_ph_UG75_DX10 = 0.2952
tau_ph_UG75_DX20 = 0.3558
tau_ph_UG100_DX10 = 0.2187
tau_ph_UG100_DX20 = 0.2584
tau_ph_UG100_DX20_NO_TURB = 0.2584

tau_char = [tau_ph_UG75_DX10 , tau_ph_UG75_DX20,
            tau_ph_UG100_DX10, tau_ph_UG100_DX20, tau_ph_UG100_DX20_NO_TURB]

# axis labels
x_label_time  = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
#y_label_xb_zb = r'$x_b/d_\mathrm{inj},~\textcolor{blue}{ z_b/d_\mathrm{inj}}$'
y_label_xb_zb = r'$x_b/d_\mathrm{inj}, z_b/d_\mathrm{inj}$'
y_label_w = r'$w/d_\mathrm{inj}$'
y_label_theta = r'$\theta~[^\circ]$'

label_mean_xb = r'$\overline{x_b}/d_\mathrm{inj}$'
label_std_xb  = r'$\sigma(x_b/d_\mathrm{inj})$'
label_mean_zb = r'$\overline{z_b}/d_\mathrm{inj}$'
label_std_zb  = r'$\sigma(z_b/d_\mathrm{inj})$'
label_mean_w  = r'$\overline{w}/d_\mathrm{inj}$'
label_std_w   = r'$\sigma(w/d_\mathrm{inj})$'
label_mean_theta  = r'$\overline{\theta}~[^\circ]$'
label_std_theta   = r'$\sigma(\theta)~[^\circ]$'

x_label_freq = r'$f~[kHz]$'
y_label_freq = r'$\mathrm{FFT}(x_b),~\mathrm{FFT}(z_b)$'



# legends
label_xb = r'$x_b/d_\mathrm{inj}$'
label_zb = r'$z_b/d_\mathrm{inj}$'
label_w  = r'$w/d_\mathrm{inj}$'

#x limits
x_lim_frequency = (0,30)

#%% Get data



r1 = np.arange(len(labels_))
barWidth = 0.50

xb = []; zb = [] 
x_width_DC = []; width_DC = []; 
y_max_DC = []; y_min_DC = []
it = []
time = []
for i in range(len(cases)):
    df = pd.read_csv(cases[i])
    it.append(df['iteration'].values)
    time.append((df['iteration'].values-1)*T*1e3/tau_char[i] + 2)
    xb.append(df['xb'].values/d_inj)
    zb.append(df['zb'].values/d_inj)
    x_width_DC.append(df['x_width_DC'].values/d_inj)
    width_DC.append(df['width_DC'].values/d_inj)
    y_max_DC.append(df['y_max_DC'].values/d_inj)
    y_min_DC.append(df['y_min_DC'].values/d_inj)

    
theta_DC = []
# For FFTs
xf_xb = []; yf_xb = []
xf_zb = []; yf_zb = []
# For mean and standard deviation
xb_mean = []; xb_std = []
zb_mean = []; zb_std = []
width_mean = []; width_std = []
y_min_mean = []; y_min_std = []
y_max_mean = []; y_max_std = []
theta_mean = []; theta_std = []
# For evolution of mean and std
xb_mean_t = []; xb_std_t = []
zb_mean_t = []; zb_std_t = []
width_mean_t = []; width_std_t = []
y_min_mean_t = []; y_min_std_t = []
y_max_mean_t = []; y_max_std_t = []
theta_mean_t = []; theta_std_t = []
for i in range(len(cases)):
    # Read data
    xi = xb[i]
    zi = zb[i]
    x_width_i = x_width_DC[i]
    width_i = width_DC[i]
    y_min_i = y_min_DC[i]
    y_max_i = y_max_DC[i]
    
    # calculate angle
    theta_1_i = np.arctan(y_max_i/x_width_i)
    theta_2_i = np.arctan(abs(y_min_i)/x_width_i)
    theta_i = (theta_1_i + theta_2_i)*180/np.pi
    theta_DC.append(theta_i)
    

    
    # Get means and stds
    xb_mean.append(np.mean(xi))
    xb_std.append(np.std(xi))
    zb_mean.append(np.mean(zi))
    zb_std.append(np.std(zi))
    width_mean.append(np.mean(width_i))
    width_std.append(np.std(width_i))
    y_min_mean.append(np.mean(y_min_i))
    y_min_std.append(np.std(y_min_i))
    y_max_mean.append(np.mean(y_max_i))
    y_max_std.append(np.std(y_max_i))
    theta_mean.append(np.mean(theta_i))
    theta_std.append(np.std(theta_i))
    
     # Get mean and std evolution graphs
    xb_mean_t_i = []; xb_std_t_i = []
    zb_mean_t_i = []; zb_std_t_i = []
    width_mean_t_i = []; width_std_t_i = []
    y_min_mean_t_i = []; y_min_std_t_i = []
    y_max_mean_t_i = []; y_max_std_t_i = []
    theta_mean_t_i = []; theta_std_t_i = []
    for j in range(len(time[i])):
        xb_mean_t_i.append(np.mean(xi[:j+1]))
        xb_std_t_i.append(np.std(xi[:j+1]))
        zb_mean_t_i.append(np.mean(zi[:j+1]))
        zb_std_t_i.append(np.std(zi[:j+1]))
        width_mean_t_i.append(np.mean(width_i[:j+1]))
        width_std_t_i.append(np.std(width_i[:j+1]))
        y_min_mean_t_i.append(np.mean(y_min_i[:j+1]))
        y_min_std_t_i.append(np.std(y_min_i[:j+1]))
        y_max_mean_t_i.append(np.mean(y_max_i[:j+1]))
        y_max_std_t_i.append(np.std(y_max_i[:j+1]))
        theta_mean_t_i.append(np.mean(theta_i[:j+1]))
        theta_std_t_i.append(np.std(theta_i[:j+1]))
    xb_mean_t.append(xb_mean_t_i)
    xb_std_t.append(xb_std_t_i)
    zb_mean_t.append(zb_mean_t_i)
    zb_std_t.append(zb_std_t_i)
    width_mean_t.append(width_mean_t_i)
    width_std_t.append(width_std_t_i)
    y_min_mean_t.append(y_min_mean_t_i)
    y_min_std_t.append(y_min_std_t_i)
    y_max_mean_t.append(y_max_mean_t_i)
    y_max_std_t.append(y_max_std_t_i)
    theta_mean_t.append(theta_mean_t_i)
    theta_std_t.append(theta_std_t_i)
    
    # Get FFT from xb
    p_xb = xi - np.mean(xi)
    N = len(p_xb)
    y_xb = fft.fft(p_xb)
    x_xb = fftfreq(N, T)[:N//2]
    y_xb = np.abs(y_xb[0:N//2])
    y_xb = y_xb/max(y_xb)
    xf_xb.append(x_xb)
    yf_xb.append(y_xb)
    
    # Get FFT from zb
    p_zb = zi - np.mean(zi)
    N = len(p_zb)
    y_zb = fft.fft(p_zb)
    x_zb = fftfreq(N, T)[:N//2]
    y_zb = np.abs(y_zb[0:N//2])
    y_zb = y_zb/max(y_zb)
    xf_zb.append(x_zb)
    yf_zb.append(y_zb)
    
width_mean = np.array(width_mean)
width_std  = np.array(width_std)
theta_mean = np.array(theta_mean)
theta_std  = np.array(theta_std)

#%% Plot signals

# Limits to separate graphs
ylims_xb_zb_temp = [(1,8), (1.8,13), (1,9.0), (1.5,15), (1.5,15)]
ylims_w_temp     = [(0,3.2), (0,2.8), (0,3), (0,3.2), (0,3.2)]

ylims_xb_zb = []; ylims_w = [] 
for i in range(len(ylims_xb_zb_temp)):
    ylims_xb_zb.append( tuple([z/d_inj for z in ylims_xb_zb_temp[i]]) )
    ylims_w.append( tuple([z/d_inj for z in ylims_w_temp[i]]) )

for i in range(len(labels_)):
    

    fig = plt.figure(figsize=figsize_)
    ax = fig.add_subplot(111)
    lns1 = ax.plot(time[i],xb[i],color='b',label=label_xb)
    lns2 = ax.plot(time[i],zb[i],color='k',label=label_zb)
    ax2 = ax.twinx()
    lns3 = ax2.plot(time[i],width_DC[i],'r',label=label_w)
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    if i==0:
        ax.legend(lns, labs, loc='upper left')
    #ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlabel(x_label_time)
    ax.set_ylabel(y_label_xb_zb)
    ax2.set_ylabel(y_label_w)
    ax.set_ylim(ylims_xb_zb[i])
    ax2.set_ylim(ylims_w[i])
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.label.set_color('red')
    plt.tight_layout()
    plt.savefig(folder_manuscript+'instant_xb_zb_w_'+save_labels[i]+'.pdf')
    plt.show()
    plt.close()


    
#%% get info correlations
    
q = 6
We_g_op1 = 7.21*100**2*d_inj/0.022; We_aero_op1 = 7.21*23.33**2*d_inj/0.022
We_g_op2 = 7.21*75**2*d_inj/0.022 ; We_aero_op2 = 7.21*15.5**2*d_inj/0.022
Re_l_op1 = 795*23.33*d_inj/(1.5e-3)
Re_l_op2 = 795*17.50*d_inj/(1.5e-3)
# Wu 1997
xb_wu_1997 = 8.06
zb_wu_1997 = 3.07*q**0.53
# Patil 2021
xb_patil_2021 = 8.6*q**(-0.4)
zb_patil_2021_op1 = 1.48*q**0.3*We_g_op1**0.1
zb_patil_2021_op2 = 1.48*q**0.3*We_g_op2**0.1
width_patil_2021_op1 = 0.1*Re_l_op1**0.46
width_patil_2021_op2 = 0.1*Re_l_op2**0.46
theta_patil_2021_op1 = 4.07*Re_l_op1**0.29
theta_patil_2021_op2 = 4.07*Re_l_op2**0.29
# Wang 2011
xb_wang_2011 = 6.9
zb_wang_2011 = 2.5*q**0.53
# Ragucci 2007
xb_ragucci_2007_op1 = 3.687*q**(-0.068)*We_aero_op1**0.420
zb_ragucci_2007_op1 = 4.355*q**0.416*We_aero_op1**0.085
xb_ragucci_2007_op2 = 3.687*q**(-0.068)*We_aero_op2**0.420
zb_ragucci_2007_op2 = 4.355*q**0.416*We_aero_op2**0.085
# 2000 Fuller
Cab = 2.58
Cd = 4.39
xb_fuller_2000 = Cd*Cab**2/np.pi
zb_fuller_2000_op1 = Cab*23.33/100*np.sqrt(795/7.21)
zb_fuller_2000_op2 = Cab*17.50/75*np.sqrt(795/7.21)




#%% Graphs with mean and std values


plt.rcParams['legend.fontsize'] = 50*FFIG

# xb bar graph
plt.figure(figsize=(20.0, 6.5))
plt.title(label_mean_xb)
plt.bar(r1, xb_mean, yerr=xb_std, width=barWidth, color='gray',edgecolor='black', label='Interior boundaries', capsize=barWidth*20)
#plt.xlabel(r'$\mathrm{Case}$')#, fontweight='bold')
plt.ylabel(label_mean_xb)
plt.xticks([r for r in range(len(labels_))], labels_)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))
#plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.pdf')
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
#plt.savefig(folder_manuscript+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# zb bar graph
plt.figure(figsize=(20.0, 6.5))
plt.title(label_mean_zb)
plt.bar(r1, zb_mean, yerr=zb_std, width=barWidth, color='gray',edgecolor='black', label='Interior boundaries', capsize=barWidth*20)
#plt.xlabel(r'$\mathrm{Case}$')#, fontweight='bold')
plt.ylabel(label_mean_zb)
plt.xticks([r for r in range(len(labels_))], labels_)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))
#plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.pdf')
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
#plt.savefig(folder_manuscript+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
plt.show()
plt.close()

# width bar graph
plt.figure(figsize=(20.0, 6.5))
plt.title(label_mean_w)
plt.bar(r1, width_mean, yerr=width_std, width=barWidth, color='gray',edgecolor='black', label='Interior boundaries', capsize=barWidth*20)
'''
plt.plot([r1[0]-barWidth/2,r1[1]+barWidth/2],[width_patil_2021_op2*d_inj*1e3]*2,'--k')
plt.plot([r1[2]-barWidth/2,r1[3]+barWidth/2],[width_patil_2021_op1*d_inj*1e3]*2,'--k')
'''
#plt.xlabel(r'Case')#, fontweight='bold')
plt.ylabel(label_mean_w)
plt.xticks([r for r in range(len(labels_))], labels_)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))
#plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.pdf')
#plt.savefig('./figures_isox/'+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
#plt.savefig(folder_manuscript+OP+'_QL_isox_bar_plot.eps',format='eps',dpi=1000)
plt.show()
plt.close()




#%% Convergence graphs

figsize_mean_convergence = (FFIG*20,FFIG*16)

# mean(xb)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], xb_mean_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], xb_mean_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], xb_mean_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], xb_mean_t[i], color='green',label=labels_[i]) 
i = 4; plt.plot(time[i], xb_mean_t[i], '--', color='green',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_mean_xb)
plt.xticks([0,5,10,15,20])
plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_xb+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_xb.pdf')
plt.show()
plt.close()

# std(xb)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], xb_std_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], xb_std_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], xb_std_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], xb_std_t[i], color='green',label=labels_[i]) 
i = 4; plt.plot(time[i], xb_std_t[i], '--', color='green',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_std_xb)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_xb+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()

# mean(zb)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], zb_mean_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], zb_mean_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], zb_mean_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], zb_mean_t[i], color='green',label=labels_[i])
i = 4; plt.plot(time[i], zb_mean_t[i], '--', color='green',label=labels_[i])  
plt.xlabel(x_label_time)
plt.ylabel(label_mean_zb)
plt.xticks([0,5,10,15,20])
#plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_zb+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_zb.pdf')
plt.show()
plt.close()

# std(zb)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], zb_std_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], zb_std_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], zb_std_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], zb_std_t[i], color='green',label=labels_[i]) 
i = 4; plt.plot(time[i], zb_std_t[i], '--', color='green',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_std_zb)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_zb+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()

# mean(width)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], width_mean_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], width_mean_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], width_mean_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], width_mean_t[i], color='green',label=labels_[i]) 
i = 4; plt.plot(time[i], width_mean_t[i], '--', color='green',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_mean_w)
plt.xticks([0,5,10,15,20])
#plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_w+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_width.pdf')
plt.show()
plt.close()

# std(width)
plt.figure(figsize=figsize_mean_convergence)
i = 0; plt.plot(time[i], width_std_t[i], color='black',label=labels_[i]) 
i = 1; plt.plot(time[i], width_std_t[i], color='grey',label=labels_[i]) 
i = 2; plt.plot(time[i], width_std_t[i], color='blue',label=labels_[i]) 
i = 3; plt.plot(time[i], width_std_t[i], color='green',label=labels_[i])
i = 4; plt.plot(time[i], width_std_t[i], '--', color='green',label=labels_[i])
plt.xlabel(x_label_time) 
plt.ylabel(label_std_w)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_w+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()




#%% plot graph zb vs xb
plt.rcParams['legend.fontsize'] = 40*FFIG

width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG

# xb, zb: scatterplot with mean values and std
fig = plt.figure(figsize=figsize_)
#plt.title(r'$\overline{x_b}~\mathrm{vs}~\overline{z_b}$')
# Lines
plt.plot([0,10*3],[0,10*3],'k',zorder=1,linewidth=4*FFIG)
#plt.text(20.0,21.1,r'$\overline{z_b} = \overline{x_b}$',rotation=38, fontsize=60*FFIG)
plt.text(9.0,9.5,r'$\overline{z_b} = \overline{x_b}$',rotation=30, fontsize=60*FFIG)
#plt.plot([0,10*3],[0,7.75*3],'--k',zorder=0) 
plt.text(2.0,11.4,r'$(\mathrm{a})$',fontsize=80*FFIG)
'''
# Experimental correlations from Wu 1997
plt.scatter(xb_wu_1997,zb_wu_1997,s=500,marker='*',label=r'$\mathrm{Wu}~1997$')
plt.errorbar(xb_wu_1997,zb_wu_1997,xerr=1.46, yerr=0.71)
# Experimental correlations from Wang 2011
plt.scatter(xb_wang_2011,zb_wang_2011,s=500,marker='*',label=r'$\mathrm{Wang}~2011$')
# Experimental correlations from Ragucci 2007
#plt.scatter(xb_ragucci_2007_op1,zb_ragucci_2007_op1,s=500,marker='*',label='Ragucci 2007 OP1')
#plt.scatter(xb_ragucci_2007_op2,zb_ragucci_2007_op2,s=500,marker='*',label='Ragucci 2007 OP2')
# Experimental correlations from Patil 2021
plt.scatter(xb_patil_2021,zb_patil_2021_op2,s=500,marker='*',color='black',label=r'$\mathrm{Patil}~2021~\mathrm{UG}75$')
plt.errorbar(xb_patil_2021,zb_patil_2021_op2,xerr=0.84, yerr=0.84,color='black')
plt.scatter(xb_patil_2021,zb_patil_2021_op1,s=500,marker='*',color='blue',label=r'$\mathrm{Patil}~2021~\mathrm{UG}100$')
plt.errorbar(xb_patil_2021,zb_patil_2021_op1,xerr=0.84,yerr=0.84, color='blue')
'''
# Numerical results
i = 0; plt.scatter(xb_mean[i], zb_mean[i], s=260, color='black',label=labels_[i]) 
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 1; plt.scatter(xb_mean[i], zb_mean[i], s=260, marker='^',color='black',label=labels_[i])
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 2; plt.scatter(xb_mean[i], zb_mean[i], s=260, color='blue',label=labels_[i])
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 3; plt.scatter(xb_mean[i], zb_mean[i], s=260, marker='^',color='blue',label=labels_[i]) 
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 4; plt.scatter(xb_mean[i], zb_mean[i], s=260, marker='^',color='red',label=labels_[i]) 
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.xticks([4,6,8,10,12])
plt.yticks([4,6,8,10,12])
plt.xlim(3.5,12)
plt.ylim(3.5,12)
plt.xlabel(r'$\overline{x_b}/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{z_b}/d_\mathrm{inj}$')
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'map_xb_zb.pdf')
plt.show()
plt.close()


# OJO: hacer barras de errores con barritas verticales

#%% plot graph width vs xb

# xb, zb: scatterplot with mean values and std
fig = plt.figure(figsize=figsize_)
plt.text(2.0,5.08,r'$(\mathrm{b})$',fontsize=80*FFIG)
'''
# Experimental correlations from Wu 1997
plt.scatter(xb_wu_1997,zb_wu_1997,s=500,marker='*',label=r'$\mathrm{Wu}~1997$')
plt.errorbar(xb_wu_1997,zb_wu_1997,xerr=1.46, yerr=0.71)
# Experimental correlations from Wang 2011
plt.scatter(xb_wang_2011,zb_wang_2011,s=500,marker='*',label=r'$\mathrm{Wang}~2011$')
# Experimental correlations from Ragucci 2007
#plt.scatter(xb_ragucci_2007_op1,zb_ragucci_2007_op1,s=500,marker='*',label='Ragucci 2007 OP1')
#plt.scatter(xb_ragucci_2007_op2,zb_ragucci_2007_op2,s=500,marker='*',label='Ragucci 2007 OP2')
# Experimental correlations from Patil 2021
plt.scatter(xb_patil_2021,zb_patil_2021_op2,s=500,marker='*',color='black',label=r'$\mathrm{Patil}~2021~\mathrm{UG}75$')
plt.errorbar(xb_patil_2021,zb_patil_2021_op2,xerr=0.84, yerr=0.84,color='black')
plt.scatter(xb_patil_2021,zb_patil_2021_op1,s=500,marker='*',color='blue',label=r'$\mathrm{Patil}~2021~\mathrm{UG}100$')
plt.errorbar(xb_patil_2021,zb_patil_2021_op1,xerr=0.84,yerr=0.84, color='blue')
'''
# Numerical results
i = 0; plt.scatter(xb_mean[i], width_mean[i], s=260, color='black',label=labels_[i]) 
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 1; plt.scatter(xb_mean[i], width_mean[i], s=260, marker='^',color='black',label=labels_[i])
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 2; plt.scatter(xb_mean[i], width_mean[i], s=260, color='blue',label=labels_[i])
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 3; plt.scatter(xb_mean[i], width_mean[i], s=260, marker='^',color='blue',label=labels_[i]) 
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 4; plt.scatter(xb_mean[i], width_mean[i], s=260, marker='^',color='red',label=labels_[i]) 
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.xticks([4,6,8,10,12])
plt.yticks([4,5])
plt.xlim(3.5,12)
plt.xlabel(r'$\overline{x_b}/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{w}/d_\mathrm{inj}$')
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'map_xb_width.pdf')
plt.show()
plt.close()
