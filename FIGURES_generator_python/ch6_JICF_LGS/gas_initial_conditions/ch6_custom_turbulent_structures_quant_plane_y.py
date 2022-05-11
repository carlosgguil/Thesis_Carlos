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
import scipy.signal as signal

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
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]
#rc('text.latex', preamble='\usepackage{color}')


figsize_u_vs_z = (FFIG*16,FFIG*30)
figsize_u_vs_x = (FFIG*25,FFIG*16)
figsize_several_in_a_row = (FFIG*55,FFIG*25)

d_inj = 0.45
T     = 1.5E-6
figsize_ = (FFIG*26,FFIG*16)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/gas_field_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/JICF/turbulence_state_u_mean_rms_probes/'
folder_ALM = folder + 'ALM_thesis/'



#%%  cases and labels

cases_dat = [folder + 'SPS_UG100_DX10/',
             folder + 'ALM_thesis/custom_inlet_no_droplets/']

cases_ICS = [folder_ALM + 'no_ALM_no_droplets/',
            folder_ALM + 'FDC_0p24_no_droplets/']
             #folder_ALM + 'FDC_0p30_no_droplets/']
cases_LGS = cases_ICS
'''
            [folder_ALM + 'no_ALM_with_droplets/',
             folder_ALM + 'ALM_initial_with_droplets/',
             folder_ALM + 'FDC_0p24_with_droplets/',
             folder_ALM + 'FDC_0p30_with_droplets/']
'''

label_u_ax  = r'$\overline{u} ~[\mathrm{m}~\mathrm{s}^{-1}$]'
label_x_ax   = '$x ~[\mathrm{mm}]$'
label_z_ax   = '$z ~[\mathrm{mm}]$'

labels_cases_dat = [r'$\mathrm{UG100\_DX10}$',
                    r'$\mathrm{Prescribed}$']
labels_cases_ICS = [r'$\mathrm{No~pert.}$', 
                    r'$\mathrm{ALM~optimal}$']
                    #r'$\mathrm{ALM~modified}$']
formats_dat = ['k','b']                    
formats_ICS = ['r', 'y']
formats_LGS = ['--r', '--y']

'''
# test actuator forces
cases_ICS = [folder_ALM + 'no_ALM_no_droplets/',
             folder_ALM + 'ALM_initial_no_droplets/',
             folder_ALM + 'ALM_final_no_droplets/',
             folder_ALM + 'FDC_0p17/',
             folder_ALM + 'FDC_0p24/',
             folder_ALM + 'FDC_0p30/',
             folder_ALM + 'FDC_0p35/']
cases_LGS = cases_ICS
labels_cases_ICS = [r'$\mathrm{No~pert.}$',
                    r'$\mathrm{ALM~baseline}$' , 
                    r'$\mathrm{ALM~optimal}$',
                    r'F = 0.17', 
                    r'F = 0.24',
                    r'F = 0.30',
                    r'F = 0.35']
formats_ICS = ['b', 'r', 'g','--k','--b','--r','--y']
formats_LGS = formats_ICS
'''

labels_x_planes = [r'$x = 1~\mathrm{mm}$',
                   r'$x = 2.5~\mathrm{mm}$',
                   r'$x = 5~\mathrm{mm}$',  
                   r'$x = 10~\mathrm{mm}$', 
                   r'$x = 15~\mathrm{mm}$']

labels_z_planes = [r'$z = 0.5~\mathrm{mm}$', 
                   r'$z = 1.6~\mathrm{mm}$', 
                   r'$z = 2~\mathrm{mm}$', 
                   r'$z = 3~\mathrm{mm}$', 
                   r'$z = 4~\mathrm{mm}$', 
                   r'$z = 5~\mathrm{mm}$'] 

#format_SPS = '-.k'

x_lim_u_vs_x = (0,20)
y_lim_u_vs_x = (-20,120)

x_ticks_u_vs_x = [3,0, 5, 10, 15, 20]
y_ticks_u_vs_x = [-20, 0, 20, 40, 60, 80, 100, 120]

#%% get SPS data

# Define arrays
z_values_x_lines      = [ [] for m in range(len(cases_dat)) ]
u_mean_values_x_lines = [ [] for m in range(len(cases_dat)) ]
v_mean_values_x_lines = [ [] for m in range(len(cases_dat)) ]
w_mean_values_x_lines = [ [] for m in range(len(cases_dat)) ]
x_values_z_lines      = [ [] for m in range(len(cases_dat)) ]
u_mean_values_z_lines = [ [] for m in range(len(cases_dat)) ]
v_mean_values_z_lines = [ [] for m in range(len(cases_dat)) ]
w_mean_values_z_lines = [ [] for m in range(len(cases_dat)) ]

for i in range(len(cases_dat)):
    
    # y plane
    case_i_folder = cases_dat[i]+'probes_turb_gas_planeY/'
            
    #----- x lines
    line_x01 = case_i_folder+'/line_planeY_x01mm_U_MEAN.dat'
    line_x02p5 = case_i_folder+'/line_planeY_x02p5mm_U_MEAN.dat'
    line_x05 = case_i_folder+'/line_planeY_x05mm_U_MEAN.dat'
    line_x10 = case_i_folder+'/line_planeY_x10mm_U_MEAN.dat'
    line_x15 = case_i_folder+'/line_planeY_x15mm_U_MEAN.dat'
    try: # cases full domain (SPS and ICS with ALM)
        f = open(line_x01, 'r')
        f.close()
        lines_x = [line_x01, line_x02p5, line_x05, line_x10, line_x15]    
        IS_CUSTOM_INLET = False
        
    except: # cases reduced domain (ICS reduced domain with custom inlet)
        lines_x = [line_x05, line_x10, line_x15]    
        IS_CUSTOM_INLET = True
    
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
        
        
    if IS_CUSTOM_INLET:
        z_values_x_lines[i].insert(0, np.nan)
        z_values_x_lines[i].insert(0, np.nan)
        u_mean_values_x_lines[i].insert(0, np.nan)
        u_mean_values_x_lines[i].insert(0, np.nan)
        v_mean_values_x_lines[i].insert(0, np.nan)
        v_mean_values_x_lines[i].insert(0, np.nan)
        w_mean_values_x_lines[i].insert(0, np.nan)
        w_mean_values_x_lines[i].insert(0, np.nan)
        '''
        z_values_x_lines[i] = np.insert(z_values_x_lines[i], 0, None)
        z_values_x_lines[i] = np.insert(z_values_x_lines[i], 0, None)
        u_mean_values_x_lines[i] = np.insert(u_mean_values_x_lines[i], 0, None)
        u_mean_values_x_lines[i] = np.insert(u_mean_values_x_lines[i], 0, None)
        v_mean_values_x_lines[i] = np.insert(v_mean_values_x_lines[i], 0, None)
        v_mean_values_x_lines[i] = np.insert(v_mean_values_x_lines[i], 0, None)
        w_mean_values_x_lines[i] = np.insert(w_mean_values_x_lines[i], 0, None)
        w_mean_values_x_lines[i] = np.insert(w_mean_values_x_lines[i], 0, None)
        '''
    
    
    
    #---- z lines
    line_z00p5 = case_i_folder+'/line_planeY_z00p5mm_U_MEAN.dat'
    #line_z01 = case_i_folder+'/line_planeY_z01mm_U_MEAN.dat'
    line_z01p6 = case_i_folder+'/line_planeY_z01p6mm_U_MEAN.dat'
    line_z02 = case_i_folder+'/line_planeY_z02mm_U_MEAN.dat'
    line_z03 = case_i_folder+'/line_planeY_z03mm_U_MEAN.dat'
    line_z04 = case_i_folder+'/line_planeY_z04mm_U_MEAN.dat'
    line_z05 = case_i_folder+'/line_planeY_z05mm_U_MEAN.dat'
    
    
    lines_z = [line_z00p5, line_z01p6, line_z02, line_z03, line_z04, line_z05]    
    
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
        
        if IS_CUSTOM_INLET:
            x_values_z_lines[i][j] = df['X'].values[1:]*1e3+ 3
            u_mean_values_z_lines[i][j] = df['U_MEAN(1)'].values[1:]
            v_mean_values_z_lines[i][j] = df['U_MEAN(2)'].values[1:]
            w_mean_values_z_lines[i][j] = df['U_MEAN(3)'].values[1:]
        else:
            x_values_z_lines[i][j] = df['X'].values*1e3
            u_mean_values_z_lines[i][j] = df['U_MEAN(1)'].values
            v_mean_values_z_lines[i][j] = df['U_MEAN(2)'].values
            w_mean_values_z_lines[i][j] = df['U_MEAN(3)'].values
            

#%% get ICS and LGS data


lines_z_along_x = ['planey0_z01p6.csv', 'planey0_z04p0.csv']
lines_x_along_z = ['planey0_x02p5.csv', 'planey0_x05p0.csv', 'planey0_x10p0.csv']

x_values_z_lines_ICS = [[] for i in range(len(cases_ICS))]
x_values_z_lines_LGS = [[] for i in range(len(cases_ICS))]
u_values_z_lines_ICS = [[] for i in range(len(cases_ICS))]
u_values_z_lines_LGS = [[] for i in range(len(cases_ICS))]
z_values_x_lines_ICS = [[] for i in range(len(cases_ICS))]
z_values_x_lines_LGS = [[] for i in range(len(cases_ICS))]
u_values_x_lines_ICS = [[] for i in range(len(cases_ICS))]
u_values_x_lines_LGS = [[] for i in range(len(cases_ICS))]
for i in range(len(cases_ICS)):
    case_ICS_i = cases_ICS[i]
    case_LGS_i = cases_LGS[i]
    for j in range(len(lines_z_along_x)):
        df_ICS = pd.read_csv(case_ICS_i+lines_z_along_x[j])
        x_values_z_lines_ICS[i].append(df_ICS['Points_0'].values*1e3)
        u_values_z_lines_ICS[i].append(df_ICS['U_MEAN_0'].values)
        
        df_LGS = pd.read_csv(case_LGS_i+lines_z_along_x[j])
        x_values_z_lines_LGS[i].append(df_LGS['Points_0'].values*1e3)
        u_values_z_lines_LGS[i].append(df_LGS['U_MEAN_0'].values)
        
        
    for j in range(len(lines_x_along_z)):
        df_ICS = pd.read_csv(case_ICS_i+lines_x_along_z[j])
        z_values_x_lines_ICS[i].append(df_ICS['Points_2'].values*1e3)
        u_values_x_lines_ICS[i].append(df_ICS['U_MEAN_0'].values)
        
        df_LGS = pd.read_csv(case_LGS_i+lines_x_along_z[j])
        z_values_x_lines_LGS[i].append(df_LGS['Points_2'].values*1e3)
        u_values_x_lines_LGS[i].append(df_LGS['U_MEAN_0'].values)


#%% Plots u vs x
u_to_plot = u_mean_values_z_lines

x_lim_u_vs_x = (0,20)
y_lim_u_vs_x = (-20,120)


# z = 1.6 mm
j = 1
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])
# SPS
i = 0; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],formats_dat[i], label=labels_cases_dat[i])
# custom (filtered)
i = 1
#plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],formats_dat[i], label=labels_cases_dat[i])

# filter shit
x_to_filter = x_values_z_lines[i][j]
u_to_filter = u_to_plot[i][j]

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.2 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

# Second, apply the filter
u_filtered = signal.filtfilt(B,A, u_to_filter)
plt.plot(x_values_z_lines[i][j],u_filtered,formats_dat[i], label=labels_cases_dat[i])


# ICS/LGS
k = 0
for i in range(len(cases_ICS)):
    plt.plot(x_values_z_lines_ICS[i][k],u_values_z_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(x_values_z_lines_LGS[i][k],u_values_z_lines_LGS[i][k],
             formats_LGS[i])
plt.xticks(x_ticks_u_vs_x)
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x)
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
#plt.legend(loc='best',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'custom_line_y0_along_x_z01p6.pdf')
plt.show()
plt.close()




#%%

# z = 4 mm
j = 4
plt.figure(figsize=figsize_u_vs_x)
plt.title(labels_z_planes[j])

# SPS
i = 0; plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],formats_dat[i], label=labels_cases_dat[i])

# custom (filtered)
i = 1
#plt.plot(x_values_z_lines[i][j],u_to_plot[i][j],formats_dat[i], label=labels_cases_dat[i])

# filter shit
x_to_filter = x_values_z_lines[i][j]
u_to_filter = u_to_plot[i][j]

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.2 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

# Second, apply the filter
u_filtered = signal.filtfilt(B,A, u_to_filter)
plt.plot(x_values_z_lines[i][j],u_filtered,formats_dat[i], label=labels_cases_dat[i])


# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    plt.plot(x_values_z_lines_ICS[i][k],u_values_z_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    plt.plot(x_values_z_lines_LGS[i][k],u_values_z_lines_LGS[i][k],
             formats_LGS[i])
plt.xticks(x_ticks_u_vs_x)
plt.yticks(y_ticks_u_vs_x)
plt.xlim(x_lim_u_vs_x)
plt.ylim(y_lim_u_vs_x) #plt.ylim((60,120))
plt.xlabel(label_x_ax)
plt.ylabel(label_u_ax)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.legend(loc='best',ncol=2)
plt.tight_layout()
plt.savefig(folder_manuscript+'custom_line_y0_along_x_z04p0.pdf')
plt.show()
plt.close()






#%% Plots u vs z

figsize_several_in_a_row = (FFIG*35,FFIG*20)

u_to_plot = u_mean_values_x_lines
'''
fig = plt.figure(figsize=figsize_several_in_a_row)
gs = fig.add_gridspec(1, 3, wspace=0)
axs = gs.subplots(sharex=False, sharey=True)
(ax1, ax2, ax3) = gs.subplots(sharey='row')
'''

fig = plt.figure(figsize=figsize_several_in_a_row)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharey = ax1)






# x = 5 mm
j = 2
# SPS
i = 0; ax1.plot(u_to_plot[i][j],z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])


# custom (filtered)
i = 1
#ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])

# filter shit
x_to_filter = z_values_x_lines[i][j]
u_to_filter = u_to_plot[i][j]

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.2 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

# Second, apply the filter
u_filtered = signal.filtfilt(B,A, u_to_filter)
ax1.plot(u_filtered,z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])



# ICS/LGS
k = 1
for i in range(len(cases_ICS)):
    ax1.plot(u_values_x_lines_ICS[i][k],z_values_x_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax1.plot(u_values_x_lines_LGS[i][k],z_values_x_lines_LGS[i][k],
             formats_LGS[i])
ax1.set_title(labels_x_planes[j])







# x = 10 mm
j = 3
# SPS
i = 0; ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])

# custom (filtered)
i = 1
#ax2.plot(u_to_plot[i][j],z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])

# filter shit
x_to_filter = z_values_x_lines[i][j]
u_to_filter = u_to_plot[i][j]

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.2 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

# Second, apply the filter
u_filtered = signal.filtfilt(B,A, u_to_filter)
ax2.plot(u_filtered,z_values_x_lines[i][j],formats_dat[i], label=labels_cases_dat[i])


# ICS/LGS
k = 2
for i in range(len(cases_ICS)):
    ax2.plot(u_values_x_lines_ICS[i][k],z_values_x_lines_ICS[i][k],
             formats_ICS[i], label=labels_cases_ICS[i])
    ax2.plot(u_values_x_lines_LGS[i][k],z_values_x_lines_LGS[i][k],
             formats_LGS[i])
ax2.set_title(labels_x_planes[j])
ax2.legend(bbox_to_anchor=(1.8,0.75))

ax1.set(ylabel = label_z_ax)
for ax in [ax1,ax2]:
    ax.label_outer()
    ax.set(xlabel=label_u_ax)
    ax.set_xlim(-10,130)
    ax.xaxis.set_ticks([0,50,100])
    #ax.grid()
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
#plt.ylabel([0,2,4,6,8, 10])
plt.ylim(0,10)
#ax2.yaxis.set_ticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.savefig(folder_manuscript+'custom_lines_y0_along_z_ux_mean.pdf')
plt.show()
plt.close()

'''
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
#plt.savefig(folder_manuscript+'ALM_lines_y0_along_z_ux_mean.pdf')
plt.show()
plt.close
'''