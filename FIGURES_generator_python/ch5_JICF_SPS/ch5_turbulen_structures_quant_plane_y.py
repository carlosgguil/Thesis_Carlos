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
figsize_u_vs_x = (FFIG*30,FFIG*16)
figsize_several_in_a_row = (FFIG*55,FFIG*25)

d_inj = 0.45
T     = 1.5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/turbulent_structures/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/turbulence_state_u_mean_rms_probes/'



#%%  cases and labels

cases = [folder + 'SPS_UG75_DX20/',
         folder + 'SPS_UG75_DX20/',
         folder + 'SPS_UG100_DX10/',
         folder + 'SPS_UG100_DX20/']

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x ~[\mathrm{mm}]$'
label_z_ax   = '$z ~[\mathrm{mm}]$'

labels_cases = [r'$\mathrm{UG75}\_\mathrm{DX10}$' ,r'$\mathrm{UG75}\_\mathrm{DX20}$',
                r'$\mathrm{UG100}\_\mathrm{DX10}$' , r'$\mathrm{UG100}\_\mathrm{DX20}$']


labels_x_planes = [r'$x = 1~\mathrm{mm}$', 
                   r'$x = 2.5~\mathrm{mm}$', 
                   r'$x = 5~\mathrm{mm}$',  
                   r'$x = 10~\mathrm{mm}$', 
                   r'$x = 15~\mathrm{mm}$']

labels_z_planes = [r'$z = 0.5~\mathrm{mm}$', 
                   r'$z = 1~\mathrm{mm}$', 
                   r'$z = 2~\mathrm{mm}$', 
                   r'$z = 3~\mathrm{mm}$', 
                   r'$z = 4~\mathrm{mm}$', 
                   r'$z = 5~\mathrm{mm}$'] 


x_lim_u_vs_x = (0,20)
y_lim_u_vs_x = (-20,120)

y_ticks_u_vs_x = [-20, 0, 20, 40, 60, 80, 100, 120]

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
    case_i_folder = cases[i]+'probes_turb_gas_planeY/'
    
        
    #----- x lines
    line_x01 = case_i_folder+'/line_planeY_x01mm_U_MEAN.dat'
    line_x02p5 = case_i_folder+'/line_planeY_x02p5mm_U_MEAN.dat'
    line_x05 = case_i_folder+'/line_planeY_x05mm_U_MEAN.dat'
    line_x10 = case_i_folder+'/line_planeY_x10mm_U_MEAN.dat'
    line_x15 = case_i_folder+'/line_planeY_x15mm_U_MEAN.dat'
    
    
           
    lines_x = [line_x01, line_x02p5, line_x05, line_x10, line_x15]    
    
    # add arrays per each plane
    z_values_x_lines[i]      = [ [] for m in range(len(lines_x)) ]
    u_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    v_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    w_mean_values_x_lines[i] = [ [] for m in range(len(lines_x)) ]
    
    for j in range(len(lines_x)):
        probe = pd.read_csv(lines_x[j],sep='(?<!\\#)\s+',engine='python')
        probe.columns =  [name.split(':')[1] for name in probe.columns] 
        
        # Get indices where time instants changes
        indices = [-1]
        for n in range(len(probe)):
            if np.isnan(probe.loc[n]['total_time']):
                indices.append(n)
                
        # recover mean values from last time instant
        df = probe.loc[indices[-2]+1:indices[-1]-1]
        
        
        z_values_x_lines[i][j] = df['Z'].values*1e3
        u_mean_values_x_lines[i][j] = df['U_MEAN(1)'].values
        v_mean_values_x_lines[i][j] = df['U_MEAN(2)'].values
        w_mean_values_x_lines[i][j] = df['U_MEAN(3)'].values
    
    
    
    
    
    
    #---- z lines
    line_z00p5 = case_i_folder+'/line_planeY_z00p5mm_U_MEAN.dat'
    line_z01 = case_i_folder+'/line_planeY_z01mm_U_MEAN.dat'
    line_z02 = case_i_folder+'/line_planeY_z02mm_U_MEAN.dat'
    line_z03 = case_i_folder+'/line_planeY_z03mm_U_MEAN.dat'
    line_z04 = case_i_folder+'/line_planeY_z04mm_U_MEAN.dat'
    line_z05 = case_i_folder+'/line_planeY_z05mm_U_MEAN.dat'
    
    
    lines_z = [line_z00p5, line_z01, line_z02, line_z03, line_z04, line_z05]    
    
    # add arrays per each plane
    x_values_z_lines[i]      = [ [] for m in range(len(lines_z)) ]
    u_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    v_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    w_mean_values_z_lines[i] = [ [] for m in range(len(lines_z)) ]
    
    for j in range(len(lines_z)):
        probe = pd.read_csv(lines_z[j],sep='(?<!\\#)\s+',engine='python')
        probe.columns =  [name.split(':')[1] for name in probe.columns] 
        
        # Get indices where time instants changes
        indices = [-1]
        for n in range(len(probe)):
            if np.isnan(probe.loc[n]['total_time']):
                indices.append(n)
                
        # recover mean values from last time instant
        df = probe.loc[indices[-2]+1:indices[-1]-1]
        
        
        x_values_z_lines[i][j] = df['X'].values*1e3
        u_mean_values_z_lines[i][j] = df['U_MEAN(1)'].values
        v_mean_values_z_lines[i][j] = df['U_MEAN(2)'].values
        w_mean_values_z_lines[i][j] = df['U_MEAN(3)'].values
    

#%% Plots u vs x
u_to_plot = u_mean_values_z_lines



# z = 2 mm
j = 2
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])
i = 0; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'k', label=labels_cases[i])
i = 1; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'--k', label=labels_cases[i])
i = 2; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'b', label=labels_cases[i])
i = 3; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'--b', label=labels_cases[i])
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_y0_along_x_z02.pdf')
plt.show()
plt.close()



# z = 5 mm
j = 5
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])
i = 0; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'k', label=labels_cases[i])
i = 1; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'--k', label=labels_cases[i])
i = 2; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'b', label=labels_cases[i])
i = 3; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'--b', label=labels_cases[i])
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig(folder_manuscript+'line_y0_along_x_z05.pdf')
plt.show()
plt.close()

# UG75_DX20
i = 2
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_cases[i])
j = 1; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'k',label=labels_z_planes[j])
j = 2; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'b',label=labels_z_planes[j])
j = 5; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],'r',label=labels_z_planes[j])
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


#%% Plot all to check

# UG75_DX20
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
i = 0; ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'k', label=labels_cases[i])
i = 1; ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--k', label=labels_cases[i])
i = 2; ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'b', label=labels_cases[i])
i = 3; ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--b', label=labels_cases[i])
#ax1.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax1.set_title(labels_x_planes[j])
ax1.yaxis.set_ticks([0,2,4,6,8, 10])

# x = 2.5 mm
j = 1
i = 0; ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'k', label=labels_cases[i])
i = 1; ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--k', label=labels_cases[i])
i = 2; ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'b', label=labels_cases[i])
i = 3; ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--b', label=labels_cases[i])
#ax2.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax2.set_title(labels_x_planes[j])
#ax2.xaxis.set_ticks(np.array([0,1,2,3])+2)
#ax2.yaxis.set_ticks([0,50,100,150,200,250,300])
#ax2.legend(loc='best',fontsize=40*FFIG)

# x = 5 mm
j = 2
i = 0; ax3.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'k', label=labels_cases[i])
i = 1; ax3.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--k', label=labels_cases[i])
i = 2; ax3.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'b', label=labels_cases[i])
i = 3; ax3.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--b', label=labels_cases[i])
#ax3.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax3.set_title(labels_x_planes[j])
#ax3.xaxis.set_ticks(np.array([0,1,2,3])+2)

# x = 10 mm
j = 3
i = 0; ax4.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'k', label=labels_cases[i])
i = 1; ax4.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--k', label=labels_cases[i])
i = 2; ax4.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'b', label=labels_cases[i])
i = 3; ax4.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--b', label=labels_cases[i])
#ax4.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax4.set_title(labels_x_planes[j])
#ax4.xaxis.set_ticks(np.array([0,1,2,3])+2)
ax4.legend(loc='best',fontsize=45*FFIG)


# x = 15 mm
j = 4
i = 0; ax5.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'k', label=labels_cases[i])
i = 1; ax5.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--k', label=labels_cases[i])
i = 2; ax5.plot(u_to_plot[i][j],z_values_x_lines[i][j], 'b', label=labels_cases[i])
i = 3; ax5.plot(u_to_plot[i][j],z_values_x_lines[i][j], '--b', label=labels_cases[i])
#ax5.text(0.0,6000,r'$\mathrm{UG}75\_\mathrm{DX}10$',fontsize=80*FFIG)
ax5.set_title(labels_x_planes[j])
#ax5.xaxis.set_ticks(np.array([0,1,2,3])+2)

axs.flat[0].set(ylabel = label_z_ax)
for ax in axs.flat:
    ax.label_outer()
    ax.set(xlabel=label_u_ax)
    ax.set_xlim(-30,130)
    ax.xaxis.set_ticks([0,50,100])
    ax.grid()
for ax in axs.flat[1:]:
    ax.spines['left'].set_linewidth(6*FFIG)
    ax.spines['left'].set_linestyle('-.')
#plt.ylabel([0,2,4,6,8, 10])
plt.ylim(0,10)
plt.tight_layout()
plt.savefig(folder_manuscript+'lines_y0_along_z_ux_mean.pdf')
plt.show()
plt.close
