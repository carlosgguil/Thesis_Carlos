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

d_inj = 0.3
T     = 5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/results_dense_core_modeling/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/DC_characterization/data_breakup_points/'


cases = [folder+'breakup_point_dx07p5.csv',
         folder+'breakup_point_dx10p0.csv',
         folder+'breakup_point_dx15p0.csv']

labels_ = [r'$\mathrm{DX07}$' ,r'$\mathrm{DX10}$', r'$\mathrm{DX15}$']
label_UG100_DX10 = r'$\mathrm{UG100}\_\mathrm{DX10}$'   

save_labels = ['DX07',  'DX10', 'DX15']

# Characteristic times to non-dimensionalize
tau_dr_DX07p5 = 0.359
tau_dr_DX10 = 0.354
tau_dr_DX15 = 0.562


tau_char = [tau_dr_DX07p5 , tau_dr_DX10, tau_dr_DX15]

# axis labels
x_label_time  = r'$t^{\prime}$' #r'$t~[\mathrm{ms}]$'
#y_label_xb_zb = r'$x_b/d_\mathrm{inj},~\textcolor{blue}{ z_b/d_\mathrm{inj}}$'
y_label_xb_zb = r'$x_b/d_\mathrm{inj}, z_b/d_\mathrm{inj}, w/d_\mathrm{inj}$'
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


# tp_ticks
tp_ticks_DX07 = [2,3]
tp_ticks_DX10 = [2,3,4,5]
tp_ticks_DX15 = [2,3,4,5,6,7]
tp_ticks = [tp_ticks_DX07,
            tp_ticks_DX10,
            tp_ticks_DX15]
# tp_lims
tp_lims_DX07 = (2,3)
tp_lims_DX10 = (2,5)
tp_lims_DX15 = (2,7)
tp_lims = [tp_lims_DX07,
           tp_lims_DX10,
           tp_lims_DX15]



#%%
# shifting times
tp_0_true_values = False

if tp_0_true_values:
    tp_0_DX07p5 = 0.9310/tau_dr_DX07p5
else:
    tp_0_DX07p5 = 2.05

# these are ~ 2 anyways
tp_0_DX10 = 0.7688/tau_dr_DX10
tp_0_DX15 = 1.1693/tau_dr_DX15 


# define maximum values for t' (obtained from ch8_nelem_plot.py)
tp_max_DX15 = 6.775423875670118 # diff of 1*tp
tp_max_DX10 = 4.88789371578306 
tp_max_DX07 = 3.9651507666425956 


# define t_min and ticks of mean, RMS evolution graphs
t_min = 2 #min(t_UG100_DX10_x05)



tp_0_cases = [tp_0_DX07p5, tp_0_DX10, tp_0_DX15]
tp_max_cases =  [tp_max_DX07, tp_max_DX10, tp_max_DX15]
        

# mean and std data case UG100_DX10
UG100_DX10_xb_mean = 5.474478038750494
UG100_DX10_xb_std  = 1.7779263656289632
UG100_DX10_zb_mean = 6.8286255436931595
UG100_DX10_zb_std  = 1.2637930228915344
UG100_DX10_width_mean = 4.627798782127323
UG100_DX10_width_std  = 0.5413789412702572

#%% Get data



r1 = np.arange(len(labels_))
barWidth = 0.50

xb = []; zb = [] 
width_DC = []; 
y_max_DC = []; y_min_DC = []
it = []
time = []; time_plot = []
for i in range(len(cases)):
    df = pd.read_csv(cases[i])
    it.append(df['iteration'].values)
    t_i = (df['iteration'].values-1)*T*1e3/tau_char[i] + 2
    time.append(t_i)
    xb.append(df['xb'].values/d_inj)
    zb.append(df['zb'].values/d_inj)
    width_DC.append(df['width_DC'].values/d_inj)
    y_max_DC.append(df['y_max_DC'].values/d_inj)
    y_min_DC.append(df['y_min_DC'].values/d_inj)
    
    # shift time
    tp_0_i = tp_0_cases[i]
    tp_max_i = tp_max_cases[i]
    m_i = (tp_max_i - tp_0_i)/(t_i[-1] - t_i[0])
    t_plot_i = m_i*(t_i - t_i[0]) + tp_0_i
    time_plot.append(t_plot_i)



    
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
# For evolution of mean and std
xb_mean_t = []; xb_std_t = []
zb_mean_t = []; zb_std_t = []
width_mean_t = []; width_std_t = []
y_min_mean_t = []; y_min_std_t = []
y_max_mean_t = []; y_max_std_t = []
for i in range(len(cases)):
    # Read data
    xi = xb[i]
    zi = zb[i]
    width_i = width_DC[i]
    y_min_i = y_min_DC[i]
    y_max_i = y_max_DC[i]
    

    

    
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
    
xb_mean = np.array(xb_mean)
zb_mean = np.array(zb_mean)
width_mean = np.array(width_mean)
width_std  = np.array(width_std)

#%% Plot signals

# xb_zb_ticks
xb_zb_ticks_DX07 = [-2,0,2,4,6,8]
xb_zb_ticks_DX10 = np.linspace(0,8,9) #[-2,0,2,4,6,8]
xb_zb_ticks_DX15 = np.linspace(0,8,9)
xb_zb_ticks = [xb_zb_ticks_DX07,
               xb_zb_ticks_DX10,
               xb_zb_ticks_DX15]

# xb_zb lims
xb_zb_lims_DX07 = (-2,8)
xb_zb_lims_DX10 = (-0.5,8)
xb_zb_lims_DX15 = (-0.5,8)
xb_zb_lims = [xb_zb_lims_DX07,
              xb_zb_lims_DX10,
              xb_zb_lims_DX15]

# w lims
w_lims_DX07 = (-2,8)
w_lims_DX10 = (-2,9)
w_lims_DX15 = (0,9)
w_lims = [w_lims_DX07,
          w_lims_DX10,
          w_lims_DX15]



for i in range(len(labels_)):
    
    if i == 0:
        continue


    fig = plt.figure(figsize=figsize_)
    ax = fig.add_subplot(111)
    lns1 = ax.plot(time_plot[i],xb[i],color='b',label=label_xb)
    lns2 = ax.plot(time_plot[i],zb[i],color='k',label=label_zb)
    ax2 = ax.twinx()
    lns3 = ax2.plot(time_plot[i],width_DC[i],'r',label=label_w)
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    if i==2:
        ax.legend(lns, labs, loc='upper left',ncol=3)
    #ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlabel(x_label_time)
    ax.set_ylabel(y_label_xb_zb)
    #ax2.set_ylabel(y_label_w)
    ax.set_ylim(xb_zb_lims[i])
    ax2.set_ylim(xb_zb_lims[i])
    ax.set_yticks(xb_zb_ticks[i])
    ax2.set_yticks([])
    #ax2.tick_params(axis='y', colors='red')
    #ax2.yaxis.label.set_color('red')
    ax.set_xticks(tp_ticks[i])
    ax.set_xlim(tp_lims[i])
    ax2.set_xlim(tp_lims[i])
    plt.tight_layout()
    plt.savefig(folder_manuscript+'instant_xb_zb_w_'+save_labels[i]+'.pdf')
    plt.show()
    plt.close()


    





#%% Convergence graphs

figsize_mean_convergence = (FFIG*20,FFIG*16)

# mean(xb)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], xb_mean_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], xb_mean_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], xb_mean_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_mean_xb)
plt.xticks(tp_ticks_DX15)
plt.ylim((0,3.5))
plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_xb+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_xb.pdf')
plt.show()
plt.close()

# std(xb)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], xb_std_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], xb_std_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], xb_std_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_std_xb)
plt.xticks(tp_ticks_DX15)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_xb+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()

# mean(zb)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], zb_mean_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], zb_mean_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], zb_mean_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_mean_zb)
plt.xticks(tp_ticks_DX15)
plt.ylim((0,6))
#plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_zb+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_zb.pdf')
plt.show()
plt.close()

# std(zb)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], zb_std_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], zb_std_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], zb_std_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_std_zb)
plt.xticks(tp_ticks_DX15)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_zb+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()

# mean(width)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], width_mean_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], width_mean_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], width_mean_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time)
plt.ylabel(label_mean_w)
plt.xticks(tp_ticks_DX15)
plt.ylim((0,1.5))
#plt.legend(loc='best')
plt.grid()
#plt.title(label_mean_w+r' $\mathrm{convergence~graph}$')
plt.tight_layout()
plt.savefig(folder_manuscript+'convergence_mean_width.pdf')
plt.show()
plt.close()

# std(width)
plt.figure(figsize=figsize_mean_convergence)
#i = 0; plt.plot(time_plot[i], width_std_t[i], color='red',label=labels_[i]) 
i = 1; plt.plot(time_plot[i], width_std_t[i], color='blue',label=labels_[i]) 
i = 2; plt.plot(time_plot[i], width_std_t[i], color='black',label=labels_[i]) 
plt.xlabel(x_label_time) 
plt.ylabel(label_std_w)
plt.xticks(tp_ticks_DX15)
plt.legend(loc='best')
plt.grid()
plt.title(label_std_w+r' $\mathrm{convergence~graph}$')
plt.show()
plt.close()




#%% plot graph zb vs xb

L = np.sqrt(xb_mean**2 + zb_mean**2)

L_dx07 = L[0]
L_dx10 = L[1]
L_dx15 = L[2]

xb_L = np.linspace(0,15,1000)

zb_L_dx07 = []; zb_L_dx10 = []; zb_L_dx15 = []; 
for i in range(len(xb_L)):
    
    zb_i_dx07 = np.sqrt(L_dx07**2 - xb_L[i]**2)
    zb_L_dx07.append(zb_i_dx07)
    
    zb_i_dx10 = np.sqrt(L_dx10**2 - xb_L[i]**2)
    zb_L_dx10.append(zb_i_dx10)
    
    zb_i_dx15 = np.sqrt(L_dx15**2 - xb_L[i]**2)
    zb_L_dx15.append(zb_i_dx15)

#plt.rcParams['legend.fontsize'] = 40*FFIG

width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG

# xb, zb: scatterplot with mean values and std
fig = plt.figure(figsize=figsize_)


#plt.plot(xb_L,zb_L_dx07,'--',color='grey',zorder=0)
#plt.plot(xb_L,zb_L_dx10,'--',color='grey',zorder=0)
#plt.text(6.8,3.8,r'$L_\mathrm{DC}/d_\mathrm{inj}= 8.9$', rotation = -40,color='grey',fontsize=60*FFIG)
#plt.plot(xb_L,zb_L_dx15,'--',color='grey',zorder=0)
#plt.text(10.5,4.3,r'$L_\mathrm{DC}/d_\mathrm{inj}= 12.5$', rotation = -45, color='grey',fontsize=60*FFIG)

#plt.title(r'$\overline{x_b}~\mathrm{vs}~\overline{z_b}$')
# Lines
plt.plot([0,10*3],[0,10*3],'k',zorder=1,linewidth=4*FFIG)
plt.text(2.1,2.5,r'$\overline{z_b} = \overline{x_b}$',rotation=35, fontsize=80*FFIG)
plt.text(-1.5,7.7,r'$(\mathrm{a})$',fontsize=80*FFIG)

# Numerical results
'''
i = 0; plt.scatter(xb_mean[i], zb_mean[i], s=260, color='red',label=labels_[i]) 
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='red',
             linewidth=width_error_lines,capsize=caps_error_lines)
'''
i = 1; plt.scatter(xb_mean[i], zb_mean[i], s=260,color='blue',label=labels_[i])
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 2; plt.scatter(xb_mean[i], zb_mean[i], s=260, color='black',label=labels_[i])
plt.errorbar(xb_mean[i], zb_mean[i], 
             xerr=xb_std[i], yerr=zb_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
# UG100_DX10
plt.scatter(UG100_DX10_xb_mean, UG100_DX10_zb_mean, marker='^', s=260, color='grey',label=label_UG100_DX10)
plt.errorbar(UG100_DX10_xb_mean, UG100_DX10_zb_mean, 
             xerr=UG100_DX10_xb_std, yerr=UG100_DX10_zb_std, color='grey',
             linewidth=width_error_lines,capsize=caps_error_lines)
#plt.xticks([4,6,8,10,12])
#plt.yticks([4,6,8,10,12])
plt.xlim(-0,8)
plt.ylim(0,8.2)
plt.yticks([0,2,4,6,8])
plt.xlabel(r'$\overline{x_b}/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{z_b}/d_\mathrm{inj}$')
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(folder_manuscript+'map_xb_zb.pdf')
plt.show()
plt.close()




#%% plot graph width vs xb

# xb, zb: scatterplot with mean values and std
fig = plt.figure(figsize=figsize_)
plt.text(-1.5,4.91,r'$(\mathrm{b})$',fontsize=80*FFIG)

# Numerical results
'''
i = 0; plt.scatter(xb_mean[i], width_mean[i], s=260, color='red',label=labels_[i]) 
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='red',
             linewidth=width_error_lines,capsize=caps_error_lines)
'''
i = 1; plt.scatter(xb_mean[i], width_mean[i], s=260, color='blue',label=labels_[i])
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='blue',
             linewidth=width_error_lines,capsize=caps_error_lines)
i = 2; plt.scatter(xb_mean[i], width_mean[i], s=260, color='black',label=labels_[i])
plt.errorbar(xb_mean[i], width_mean[i], 
             xerr=xb_std[i], yerr=width_std[i], color='black',
             linewidth=width_error_lines,capsize=caps_error_lines)
# UG100_DX10
plt.scatter(UG100_DX10_xb_mean, UG100_DX10_width_mean, s=260,marker='^', color='grey',label=label_UG100_DX10)
plt.errorbar(UG100_DX10_xb_mean, UG100_DX10_width_mean, 
             xerr=UG100_DX10_xb_std, yerr=UG100_DX10_width_std, color='grey',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot([-10,10],[1]*2,'--',color='grey')
#plt.text(4.2,1.2,r'$w/d_\mathrm{inj} = 1$',color='grey',fontsize=70*FFIG)
#plt.xticks([4,6,8,10,12])
plt.xlim(0,8)
plt.ylim(0,5.5)
plt.yticks([0,1,2,3,4,5])
plt.xlabel(r'$\overline{x_b}/d_\mathrm{inj}$')
plt.ylabel(r'$\overline{w}/d_\mathrm{inj}$')
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'map_xb_width.pdf')
plt.show()
plt.close()

#%% Frequencies
'''
for i in range(len(labels_)):
    plt.figure(figsize=figsize_mean_convergence)
    plt.title(labels_[i])
    plt.plot(xf_xb[i]/1000, yf_xb[i], color='black',label='xb')
    #plt.plot(xf_zb[i]/1000, yf_zb[i], color='blue',label='zb')
    plt.xlabel(r'$f~[kHz]$') 
    plt.ylabel(r'$\mathrm{FFT}$')
    plt.xlim((0,20))
    plt.legend(loc='best')
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()
'''


figsize_FFTs = figsize_ #(FFIG*20,FFIG*12) 
plt.figure(figsize=figsize_FFTs)
i = 0; plt.plot(xf_xb[i]/1000, yf_xb[i], color='blue',label=labels_[i])
i = 1; plt.plot(xf_zb[i]/1000, yf_zb[i], color='black',label=labels_[i])
i = 2; plt.plot(xf_zb[i]/1000, yf_zb[i], color='red',label=labels_[i])
#plt.plot(xf_zb[i]/1000, yf_zb[i], color='blue',label='zb')
plt.xlabel(r'$f~[kHz]$') 
plt.ylabel(r'$\mathrm{FFT} (x_b)$')
plt.xlim((0,20))
plt.xticks([0,5,10,15,20])
plt.grid()
plt.legend(fontsize=50*FFIG)
plt.tight_layout()
plt.savefig(folder_manuscript+'FFTs_BIMER.pdf')
plt.show()
plt.close()





# find frequencies for tau_str
print('FREQUENCIES')
for i in range(len(labels_)):
    if i == 1:
        index = np.where(yf_zb[i] == max(yf_zb[i]))
        index = index[0][0]
        freq = xf_zb[i][index]
    else:
        index = np.where(yf_xb[i] == max(yf_xb[i]))
        index = index[0][0]
        freq = xf_xb[i][index]
    
    tau_str = 1/freq*1e6
    print('  '+save_labels[i]+f': f = {freq} Hz, tau_str = {tau_str} micros')
    





