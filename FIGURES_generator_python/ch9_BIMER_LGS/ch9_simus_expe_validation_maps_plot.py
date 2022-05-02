# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:38:38 2021

@author: d601630
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from matplotlib import gridspec

# Change size of figures 
FFIG = 0.5

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 60*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 60*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 20*FFIG
plt.rcParams['axes.titlesize']  = 60*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 30*FFIG
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams['text.usetex'] = True
#rc('text.latex', preamble='\usepackage{color}')






folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch9_lagrangian/simus_expe_validation_temp/'
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/postprocessing/donnees_expe_Renaud/image_processing_with_matlab/csv_output_files/'
folder_simus = 'C:/Users/Carlos Garcia/Desktop/Ongoing/BIMER/LGS_simus/data_droplets_BIMER_LGS/'

# Maps N levels
N_LEVELS = 51

# Map coordinates
x_min = 10
x_max = 35
y_min = -43.5
y_max = 46.5

DX = x_max - x_min
DY = y_max - y_min

# Define figsize
AR = DX/DY
figsize_ = (FFIG*18*AR,FFIG*16)
figsize_maps_subplots = (5*FFIG*18*AR,FFIG*12)


pad_title_maps = 20

# labels
x_label_ = r'$x~[\mathrm{mm}]$'
y_label_ = r'$y~[\mathrm{mm}]$'

label_SMD = r'$SMD~[\mu \mathrm{m}]$'
label_u_axial = r'$u~[\mathrm{m~s}^{-1}]$'
label_u_vertical = r'$v~[\mathrm{m~s}^{-1}]$'

titles = [r'$\mathrm{Expe}$',r'$\mathrm{Baseline}$', 
          r'$\mathrm{ALM}$', r'$\mathrm{Evap}$']

# simus folder
cases_simus = ['baseline_no_ALM_no_evap','with_ALM_no_evap','no_ALM_with_evap']

gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.25])


# interesting for subplots formatting
labelpad_ = 0
labelpad_SMD = 25
w_space_subplots = 0.5




#%% Read expe maps

# SMD
df_SMD = pd.read_csv(folder_expe+'SMD_values_processed_matlab.csv')
data_SMD = pd.DataFrame.to_numpy(df_SMD)
# y and z coordinates (pixels numbers)
x_values_SMD = np.linspace(1,len(df_SMD.columns),len(df_SMD.columns))
y_values_SMD = np.linspace(1,len(df_SMD),len(df_SMD))

PX_SMD = x_values_SMD[-1]
PY_SMD = y_values_SMD[-1]
# transformation to physical coordinates
x_values_SMD = DX/2/((PX_SMD-1)/2)*x_values_SMD + x_min - DX/2/((PX_SMD-1)/2)
y_values_SMD = DY/(PY_SMD-1)*y_values_SMD+y_min-DY/(PY_SMD-1)
xx_values_SMD, yy_values_SMD = np.meshgrid(x_values_SMD, y_values_SMD)




# u_axial
df_u_axial = pd.read_csv(folder_expe+'u_axial_values_processed_matlab.csv')
data_u_axial = pd.DataFrame.to_numpy(df_u_axial)
# y and z coordinates (pixels numbers)
x_values_u_axial  = np.linspace(1,len(df_u_axial.columns),len(df_u_axial.columns))
y_values_u_axial  = np.linspace(1,len(df_u_axial),len(df_u_axial))

PX_u_axial  = x_values_u_axial[-1]
PY_u_axial  = y_values_u_axial[-1]
# transformation to physical coordinates
x_values_u_axial = DX/2/((PX_u_axial-1)/2)*x_values_u_axial + x_min - DX/2/((PX_u_axial-1)/2)
y_values_u_axial = DY/(PY_u_axial-1)*y_values_u_axial+y_min-DY/(PY_u_axial-1)
xx_values_u_axial, yy_values_u_axial = np.meshgrid(x_values_u_axial, y_values_u_axial)




# u_vertical
df_u_vertical = pd.read_csv(folder_expe+'u_vertical_values_processed_matlab.csv')
data_u_vertical = pd.DataFrame.to_numpy(df_u_vertical)
# y and z coordinates (pixels numbers)
x_values_u_vertical  = np.linspace(1,len(df_u_vertical.columns),len(df_u_vertical.columns))
y_values_u_vertical  = np.linspace(1,len(df_u_vertical),len(df_u_vertical))

PX_u_vertical  = x_values_u_vertical[-1]
PY_u_vertical  = y_values_u_vertical[-1]
# transformation to physical coordinates
x_values_u_vertical = DX/2/((PX_u_vertical-1)/2)*x_values_u_vertical + x_min - DX/2/((PX_u_vertical-1)/2)
y_values_u_vertical = DY/(PY_u_vertical-1)*y_values_u_vertical+y_min-DY/(PY_u_vertical-1)
xx_values_u_vertical, yy_values_u_vertical = np.meshgrid(x_values_u_vertical, y_values_u_vertical)



# levels calculation
levels_map_SMD = np.linspace(9,34,51)
levels_map_u_axial = np.linspace(-23,53,77)
levels_map_u_vertical = np.linspace(-17,17,34)


#%% read numerical maps

xx_all_cases = [xx_values_SMD]
yy_all_cases = [yy_values_SMD]
SMD_to_plot_all_cases = [data_SMD]
u_to_plot_all_cases = [data_u_axial]
v_to_plot_all_cases = [data_u_vertical]
w_to_plot_all_cases = [[]]

for k in range(len(cases_simus)):
    folder = folder_simus + cases_simus[k]

    with open(folder+'./pickle_map_data_no_grid', 'rb') as f:
        dict_spray = pickle.load(f)
        xx = dict_spray['xx']
        yy = dict_spray['yy']
        SMD_to_plot = dict_spray['SMD_to_plot']
        u_to_plot = dict_spray['u_to_plot']
        v_to_plot = dict_spray['v_to_plot']
        w_to_plot = dict_spray['w_to_plot']
        u_VW_to_plot = dict_spray['u_VW_to_plot']
        v_VW_to_plot = dict_spray['v_VW_to_plot']
        w_VW_to_plot = dict_spray['w_VW_to_plot']
    
    xx_all_cases.append(xx)
    yy_all_cases.append(yy)
    SMD_to_plot_all_cases.append(SMD_to_plot)
    u_to_plot_all_cases.append(u_to_plot)
    v_to_plot_all_cases.append(v_to_plot)
    w_to_plot_all_cases.append(w_to_plot)
    '''
    u_to_plot_all_cases.append(u_VW_to_plot)
    v_to_plot_all_cases.append(v_VW_to_plot)
    w_to_plot_all_cases.append(w_VW_to_plot)
    '''
        

#%% plot global SMDs


print(' ---------- Global SMDs --------')
for n in range(len(SMD_to_plot_all_cases)):
    SMD_n = SMD_to_plot_all_cases[n]
    
    SMD = 0; count = 0
    for i in range(len(SMD_n)):
        for j in range(len(SMD_n[i])):
            SMD_ij = SMD_n[i][j]
            if not np.isnan(SMD_ij):
                SMD += SMD_ij
                count += 1
    SMD = SMD/count
    print('   Case'+titles[n]+f' SMD = {SMD}')

#%% Plot individual expe maps




i = 0

dpi_ = 120

# SMD
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_SMD, yy_values_SMD, data_SMD, 
                   vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
#plt.colorbar(format = '$%d$',ticks=levels_map_SMD[::10])
#plt.title(label_SMD, pad=pad_title_maps)
plt.title(titles[i], pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
#plt.savefig(folder_manuscript+'expe_SMD_map.png',bbox_inches='tight',dpi=dpi_)
#plt.show()
plt.close()



# u axial
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_u_axial, yy_values_u_axial, data_u_axial,
                   vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
#plt.colorbar(format = '$%d$',ticks=[-20,0,20,40])
#plt.title(label_u_axial, pad=pad_title_maps)
plt.title(titles[i], pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
#plt.savefig(folder_manuscript+'expe_u_axial_map.png',bbox_inches='tight',dpi=dpi_)
#plt.show()
plt.close()

# u vertical
plt.figure(figsize=(figsize_))
plt.pcolor(xx_values_u_vertical, yy_values_u_vertical, data_u_vertical,
                   vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
#plt.colorbar(format = '$%d$',ticks=[-10,0,10])
#plt.title(label_u_vertical, pad=pad_title_maps)
plt.title(titles[i], pad=pad_title_maps)
plt.xlabel(x_label_)
plt.ylabel(y_label_)
#plt.tight_layout()
#plt.savefig(folder_manuscript+'expe_u_vertical_map.png',bbox_inches='tight',dpi=dpi_)
#plt.show()
plt.close()





#%% subplots maps SMD

fig = plt.figure(figsize=figsize_maps_subplots)


ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax5 = plt.subplot(gs[4])



# expe
i = 0
ax1.pcolor(xx_all_cases[i], yy_all_cases[i], SMD_to_plot_all_cases[i],
           vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
#ax1.set_xlabel(x_label_)
ax1.set_title(titles[i], pad=pad_title_maps)

# baseline case
i = 1
ax2.pcolor(xx_all_cases[i], yy_all_cases[i], SMD_to_plot_all_cases[i],
           vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
#ax2.set_xlabel(x_label_)
ax2.set_title(titles[i], pad=pad_title_maps)

# ALM
i = 2
ax3.pcolor(xx_all_cases[i], yy_all_cases[i], SMD_to_plot_all_cases[i],
           vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
#ax3.set_xlabel(x_label_)
ax3.set_title(titles[i], pad=pad_title_maps)

# evap
i = 3
im = ax4.pcolor(xx_all_cases[i], yy_all_cases[i], SMD_to_plot_all_cases[i],
           vmin = levels_map_SMD[0], vmax = levels_map_SMD[-1], cmap = 'jet')
#ax4.set_xlabel(x_label_)
ax4.set_title(titles[i], pad=pad_title_maps)
#ax4.set_colorbar(format = '$%d$',ticks=levels_map_SMD[::10])


cax = ax5
#cax = fig.add_axes([1.0, 0.150, 0.05, 0.75])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
plt.colorbar(im, cax = cax, format = '$%d$',ticks=[10,15,20,25,30])
cbar.set_label(label_SMD, labelpad=labelpad_SMD,fontsize=60*FFIG)


ax1.set(ylabel = y_label_)
for ax in [ax1,ax2,ax3,ax4]:
    ax.label_outer()
    ax.set(xlabel=x_label_)
    #ax.set_xlim(x_lim_u_vs_z)
    #ax.xaxis.set_ticks(x_ticks_u_vs_z)
    #ax.grid()
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
#plt.ylabel([0,2,4,6,8, 10])
#plt.ylim(y_lim_u_vs_z)
#ax2.yaxis.set_ticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=w_space_subplots)
plt.savefig(folder_manuscript+'subplots_maps_SMD.png')
plt.show()
plt.close()


#%% subplots axial velocity



fig = plt.figure(figsize=figsize_maps_subplots)


ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax5 = plt.subplot(gs[4])


# expe
i = 0
ax1.pcolor(xx_values_u_axial, yy_values_u_axial, u_to_plot_all_cases[i],
           vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
#ax1.set_xlabel(x_label_)
ax1.set_title(titles[i], pad=pad_title_maps)

# baseline case
i = 1
ax2.pcolor(xx_all_cases[i], yy_all_cases[i], u_to_plot_all_cases[i],
           vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
#ax2.set_xlabel(x_label_)
ax2.set_title(titles[i], pad=pad_title_maps)

# ALM
i = 2
ax3.pcolor(xx_all_cases[i], yy_all_cases[i], u_to_plot_all_cases[i],
           vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
#ax3.set_xlabel(x_label_)
ax3.set_title(titles[i], pad=pad_title_maps)

# evap
i = 3
im = ax4.pcolor(xx_all_cases[i], yy_all_cases[i], u_to_plot_all_cases[i],
           vmin = levels_map_u_axial[0], vmax = levels_map_u_axial[-1], cmap = 'jet')
#ax4.set_xlabel(x_label_)
ax4.set_title(titles[i], pad=pad_title_maps)
#ax4.set_colorbar(format = '$%d$',ticks=levels_map_SMD[::10])

cax = ax5
#cax = fig.add_axes([1.0, 0.150, 0.05, 0.75])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
plt.colorbar(im, cax = cax, format = '$%d$',ticks=[-20,0,20,40])
cbar.set_label(label_u_axial, labelpad=labelpad_)#,fontsize=70*FFIG)



ax1.set(ylabel = y_label_)
for ax in [ax1,ax2,ax3,ax4]:
    ax.label_outer()
    ax.set(xlabel=x_label_)
    #ax.set_xlim(x_lim_u_vs_z)
    #ax.xaxis.set_ticks(x_ticks_u_vs_z)
    #ax.grid()
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
#plt.ylabel([0,2,4,6,8, 10])
#plt.ylim(y_lim_u_vs_z)
#ax2.yaxis.set_ticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=w_space_subplots)
plt.savefig(folder_manuscript+'subplots_maps_axial_velocity.png')
plt.show()
plt.close()

#%% subplots vertical velocity




fig = plt.figure(figsize=figsize_maps_subplots)


'''
ax1 = plt.subplot(151)
ax2 = plt.subplot(152)
ax3 = plt.subplot(153)
ax4 = plt.subplot(154)
ax5 = plt.subplot(155)
'''

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax5 = plt.subplot(gs[4])




# expe
i = 0
ax1.pcolor(xx_values_u_vertical, yy_values_u_vertical, v_to_plot_all_cases[i],
           vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
#ax1.set_xlabel(x_label_)
ax1.set_title(titles[i], pad=pad_title_maps)

# baseline case
i = 1
ax2.pcolor(xx_all_cases[i], yy_all_cases[i], v_to_plot_all_cases[i],
           vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
#ax2.set_xlabel(x_label_)
ax2.set_title(titles[i], pad=pad_title_maps)

# ALM
i = 2
ax3.pcolor(xx_all_cases[i], yy_all_cases[i], v_to_plot_all_cases[i],
           vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
#ax3.set_xlabel(x_label_)
ax3.set_title(titles[i], pad=pad_title_maps)

# evap
i = 3
im = ax4.pcolor(xx_all_cases[i], yy_all_cases[i], v_to_plot_all_cases[i],
           vmin = levels_map_u_vertical[0], vmax = levels_map_u_vertical[-1], cmap = 'jet')
#ax4.set_xlabel(x_label_)
ax4.set_title(titles[i], pad=pad_title_maps)
#ax4.set_colorbar(format = '$%d$',ticks=levels_map_SMD[::10])

# colorbar
'''
cax = fig.add_axes([1.0, 0.150, 0.05, 0.75])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
plt.colorbar(im, cax = cax, format = '$%d$',ticks=[-10,0,10])
cbar.set_label(label_u_vertical, labelpad=-10)#,fontsize=70*FFIG)
#ax5.figure.set_size_inches(figsize_maps_subplots[0]*0.1, 
#                           figsize_maps_subplots[1])
'''

cax = ax5
#cax = fig.add_axes([1.0, 0.150, 0.05, 0.75])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
plt.colorbar(im, cax = cax, format = '$%d$',ticks=[-10,0,10])
cbar.set_label(label_u_vertical, labelpad=labelpad_)#,fontsize=70*FFIG)




ax1.set(ylabel = y_label_)
for ax in [ax1,ax2,ax3,ax4]:
    ax.label_outer()
    ax.set(xlabel=x_label_)
    #ax.set_xlim(x_lim_u_vs_z)
    #ax.xaxis.set_ticks(x_ticks_u_vs_z)
    #ax.grid()
    #ax.grid()
#plt.ylabel([0,2000,4000,6000,8000])
#plt.ylabel([0,2,4,6,8, 10])
#plt.ylim(y_lim_u_vs_z)
#ax2.yaxis.set_ticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=w_space_subplots)
plt.savefig(folder_manuscript+'subplots_maps_vertical_velocity.png')
plt.show()
plt.close()
