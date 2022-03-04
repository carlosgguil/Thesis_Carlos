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
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]
#rc('text.latex', preamble='\usepackage{color}')


figsize_u_vs_z = (FFIG*16,FFIG*30)
figsize_u_vs_x = (FFIG*29,FFIG*16)
figsize_several_in_a_row = (FFIG*55,FFIG*25)

d_inj = 0.45
T     = 1.5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/turbulent_structures/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/turbulence_state_u_mean/'



#%%  cases and labels

cases = [folder + 'DX07p5_lines/',
         folder + 'DX10p0_lines/',
         folder + 'DX15p0_lines/',
         folder + 'lines_u_mean_ics/']

label_u_ax  = r'$\overline{u}_c ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x_c ~[\mathrm{mm}]$'
label_z_ax   = '$z_c ~[\mathrm{mm}]$'

labels_cases = [r'$\mathrm{DX07}$' ,r'$\mathrm{DX10}$', 
                r'$\mathrm{DX15}$',r'$\mathrm{No~jet}$']
format_cases = ['k','b','r','--k']

labels_x_planes = [r'$x = 0.5~\mathrm{mm}$', 
                   r'$x = 1~\mathrm{mm}$', 
                   r'$x = 2~\mathrm{mm}$', 
                   r'$x = 4~\mathrm{mm}$',  
                   r'$x = 5~\mathrm{mm}$']

labels_z_planes = [r'$z = 0.25~\mathrm{mm}$',
                   r'$z = 0.3~\mathrm{mm}$', 
                   r'$z = 0.5~\mathrm{mm}$', 
                   r'$z = 0.75~\mathrm{mm}$',
                   r'$z = 0.8~\mathrm{mm}$', 
                   r'$z = 1~\mathrm{mm}$', 
                   r'$z = 1.5~\mathrm{mm}$'] 

# u vs x
x_lim_u_vs_x = (0,30)
y_lim_u_vs_x = (-5,70)
y_ticks_u_vs_x = [0,20,40,60,80]

# u vs z
y_lim_u_vs_z = (0,5)
x_lim_u_vs_z = (0,70)
x_ticks_u_vs_z =  [0,20,40,60]

#%% get data

# Define arrays
z_values_x_lines      = [ [] for m in range(len(cases)) ]
u_mean_values_x_lines = [ [] for m in range(len(cases)) ]
v_mean_values_x_lines = [ [] for m in range(len(cases)) ]
w_mean_values_x_lines = [ [] for m in range(len(cases)) ]
x_values_z_lines      = [ [] for m in range(len(cases)) ]
u_mean_values_z_lines = [ [] for m in range(len(cases)) ]
v_mean_values_z_lines = [ [] for m in range(len(cases)) ]
w_mean_values_z_lines = [ [] for m in range(len(cases)) ]

for i in range(len(cases)):
    
    # y plane
    case_i_folder = cases[i]+'plane_yc/'
    
        
    #----- x lines
    line_x00p5 = case_i_folder+'/x_0p5mm.csv'
    line_x01 = case_i_folder+'/x_1p0mm.csv'
    line_x02 = case_i_folder+'/x_2p0mm.csv'
    line_x04 = case_i_folder+'/x_4p0mm.csv'
    line_x05 = case_i_folder+'/x_5p0mm.csv'
    
    
           
    lines_x = [line_x00p5, line_x01, line_x02, line_x04, line_x05]    
    
    # add arrays per each plane
    z_values_x_lines[i]      = [ [] for m in range(len(lines_x)) ]
    u_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    v_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    w_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    
    for j in range(len(lines_x)):
        
        df = pd.read_csv(lines_x[j])
        z_values_x_lines[i][j] = df['Points_2'].values*1e3
        u_mean_values_x_lines[i][j] = df['U_MEAN_0'].values
        v_mean_values_x_lines[i][j] = df['U_MEAN_1'].values
        w_mean_values_x_lines[i][j] = df['U_MEAN_2'].values

    
    #---- z lines
    line_z0p25 = case_i_folder+'/z_0p25mm.csv'
    line_z0p3 = case_i_folder+'/z_0p3mm.csv'
    line_z0p5 = case_i_folder+'/z_0p5mm.csv'
    line_z0p75 = case_i_folder+'/z_0p75mm.csv'
    line_z0p8 = case_i_folder+'/z_0p8mm.csv'
    line_z1p0 = case_i_folder+'/z_1p0mm.csv'
    line_z1p5 = case_i_folder+'/z_1p5mm.csv'
    
    
    lines_z = [line_z0p25, line_z0p3, line_z0p5, line_z0p75, 
               line_z0p8, line_z1p0, line_z1p5]    
    
    # add arrays per each plane
    x_values_z_lines[i]      = [ [] for m in range(len(lines_z)) ]
    u_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    v_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    w_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    
    for j in range(len(lines_z)):
        
        
        df = pd.read_csv(lines_z[j])
        x_values_z_lines[i][j] = df['Points_0'].values*1e3
        u_mean_values_z_lines[i][j] = df['U_MEAN_0'].values
        v_mean_values_z_lines[i][j] = df['U_MEAN_1'].values
        w_mean_values_z_lines[i][j] = df['U_MEAN_2'].values
    

#%% Plots u vs x
u_to_plot = u_mean_values_z_lines



# 
j =0
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])
plt.plot(x_lim_u_vs_x,(0,0),'k',zorder=-1,linewidth=6*FFIG)
for i in range(len(labels_cases)):
    plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],format_cases[i], label=labels_cases[i])
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='upper right',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'line_y0_along_x_zlow.pdf')
plt.show()
plt.close()



# 
j = 1
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])
plt.plot(x_lim_u_vs_x,(0,0),'k',zorder=-1,linewidth=6*FFIG)
for i in range(len(labels_cases)):
    plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],format_cases[i], label=labels_cases[i])
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_y0_along_x_zhigh.pdf')
plt.show()
plt.close()

# DX07p5
i = 2
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_cases[i])
j = 1; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'k',label=labels_z_planes[j])
j = 2; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'b',label=labels_z_planes[j])
j = 5; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'r',label=labels_z_planes[j])
#plt.xticks([4,6,8,10,12])
#plt.yticks([4,6,8,10,12])
plt.xlim(0,10)
#plt.ylim(3.5,12)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig(folder_manuscript+'map_xb_zb.pdf')
plt.show()
plt.close()


#%% Plot all to check

# DX10
i = 1
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_cases[i])
for j in range(len(labels_z_planes)):
   plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],label=labels_z_planes[j])
#plt.xticks([4,6,8,10,12])
#plt.yticks([4,6,8,10,12])
plt.xlim(0,20)
#plt.ylim(3.5,12)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig(folder_manuscript+'map_xb_zb.pdf')
plt.show()
plt.close()


#%% Plots u vs z

figsize_several_in_a_row = (FFIG*50,FFIG*20)

u_to_plot = u_mean_values_x_lines

fig = plt.figure(figsize=figsize_several_in_a_row)
gs = fig.add_gridspec(1, 5, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3, ax4, ax5) = gs.subplots(sharey='row')

# x = 1 mm
j = 0
for i in range(len(labels_cases)):
    ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j], format_cases[i], label=labels_cases[i])
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(labels_x_planes[j])
#ax1.yaxis.set_ticks([0,2,4,6,8, 10])
ax1.yaxis.set_ticks([0,1,2,3,4,5])
ax1.yaxis.set_ticks([0,1,2,3,4,5])

# x = 2.5 mm
j = 1
for i in range(len(labels_cases)):
    ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j], format_cases[i], label=labels_cases[i])
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax2.set_title(labels_x_planes[j])
#ax2.xaxis.set_ticks(np.array([0,1,2,3])+2)
#ax2.yaxis.set_ticks([0,50,100,150,200,250,300])
#ax2.legend(loc='best',fontsize=40*FFIG)

# x = 5 mm
j = 2
for i in range(len(labels_cases)):
    ax3.plot(u_to_plot[i][j],z_values_x_lines[i][j], format_cases[i], label=labels_cases[i])
#ax3.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(labels_x_planes[j])
#ax3.xaxis.set_ticks(np.array([0,1,2,3])+2)

# x = 10 mm
j = 3
for i in range(len(labels_cases)):
    ax4.plot(u_to_plot[i][j],z_values_x_lines[i][j], format_cases[i], label=labels_cases[i])
#ax4.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax4.set_title(labels_x_planes[j])
#ax4.xaxis.set_ticks(np.array([0,1,2,3])+2)
ax4.legend(loc='best',fontsize=45*FFIG)

# x = 10 mm
j = 4
for i in range(len(labels_cases)):
    ax5.plot(u_to_plot[i][j],z_values_x_lines[i][j], format_cases[i], label=labels_cases[i])
#ax4.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax5.set_title(labels_x_planes[j])
#ax4.xaxis.set_ticks(np.array([0,1,2,3])+2)
ax5.legend(loc='best',fontsize=45*FFIG)




axs.flat[0].set(ylabel = label_z_ax)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=label_u_ax)
    ax.set_xlim(x_lim_u_vs_z)
    ax.xaxis.set_ticks(x_ticks_u_vs_z)
    ax.grid()
for ax in axs.flat[1:]:
    ax.spines['left'].set_linewidth(6*FFIG)
    ax.spines['left'].set_linestyle('-.')
#plt.ylabel([0,2,4,6,8, 10])
plt.ylim(y_lim_u_vs_z)
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_y0_along_z_ux_mean.pdf')
plt.show()
plt.close
