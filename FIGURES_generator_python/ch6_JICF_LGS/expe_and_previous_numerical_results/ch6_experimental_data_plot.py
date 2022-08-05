
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from functions_expe_comparison import get_SMD_from_integrated_profile

# Change size of figures 
FFIG = 0.5

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 80*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 80*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 80*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 50*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  6*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['lines.markersize'] = 30*FFIG
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#rc('text.latex', preamble='\usepackage{color}')






folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/expe_results/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'

figsize_ = (FFIG*24,FFIG*16)
figsize_maps = (FFIG*15,FFIG*15)
figsize_maps_subplots = (FFIG*35,FFIG*22)




#%% Define cases, plot parameters


case_UG100 = '1210_01' 
case_UG75 =  '2310_01_u75' 

labels_cases = [r'UG100', r'UG75']

cases = [folder+case_UG100, folder+case_UG75]

y_label_   = r'$y~[\mathrm{mm}]$'
z_label_   = r'$z~[\mathrm{mm}]$'
label_SMD  = r'$SMD~[\mu \mathrm{m}]$'
#label_flux = r'$\mathrm{Volume~Flux}~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'
label_flux = r'$q_l~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'

label_SMD_aver  = r'$\langle SMD \rangle ~[\mu \mathrm{m}]$'
label_flux_aver = r'$\langle q_l \rangle ~[\mathrm{cm}^3 ~ \mathrm{s}^{-1} ~ \mathrm{cm}^{-2}]$'

# Map pads for colorbar and title
pad_cmap_maps  = 0.9
pad_title_maps = -120


#%% Get maps

yy_values_all  = []; zz_values_all = []
SMD_values_all = []; flux_values_all = []
SMD_global_cases = []; SMD_global_flux_weighted_cases = []
for i in range(len(cases)):
    df = pd.read_csv(cases[i]+'.dat',sep='(?<!\\#)\s+',skiprows=3,engine='python')      
                                        
    y_values = np.array(list(set(df['# Y'])))
    y_values = np.sort(y_values)
    z_values = np.array(list(set(df['Z'])))




    NP_y = len(y_values)
    NP_z = len(z_values)
    SMD_values  = np.ones([NP_z, NP_y])*np.nan
    flux_values = np.ones([NP_z, NP_y])*np.nan
    SMD_global = 0; counter_SMD_probes = 0
    SMD_global_weighted_flux = 0; flux_acc = 0
    for i in range(len(df)):
        y_i    = df['# Y'][i]
        z_i    = df['Z'][i]
        SMD_i  = df['D32'][i]
        flux_i = df['uFlx'][i]
        
        m = np.where(np.array(y_values) == y_i)[0][0]
        n = np.where(np.array(z_values) == z_i)[0][0]
        
        if SMD_i > 0:
            SMD_values[n][m] = SMD_i
            SMD_global += SMD_i
            SMD_global_weighted_flux += SMD_i*flux_i
            flux_acc += flux_i
            counter_SMD_probes += 1
        if flux_i >= 0:
            flux_values[n][m] = flux_i
        
    SMD_global_cases.append(SMD_global/counter_SMD_probes)
    SMD_global_flux_weighted_cases.append(SMD_global_weighted_flux/flux_acc)
    
    
    # Create grid
    yy_values, zz_values = np.meshgrid(y_values, z_values)
    
    
    # Append values
    yy_values_all.append(yy_values)
    zz_values_all.append(zz_values)
    SMD_values_all.append(SMD_values)
    flux_values_all.append(flux_values)
    
#%% Read spatially integrated data

y_int_values_all = []; z_int_values_all = []
flux_y_int_values_all = []; SMD_y_int_values_all = []
flux_z_int_values_all = []; SMD_z_int_values_all = []
for i in range(len(cases)):
    df_y = pd.read_csv(cases[i]+'_data_integrated_y_exp.csv')
    df_z = pd.read_csv(cases[i]+'_data_integrated_z_exp.csv')
    
    y_int_values_all.append(df_z['y_values'].values)
    flux_y_int_values_all.append(df_z['flux_y_exp'].values)
    SMD_y_int_values_all.append(df_z['SMD_y_exp'].values)
    
    z_int_values_all.append(df_y['z_values'].values)
    flux_z_int_values_all.append(df_y['flux_z_exp'].values)
    SMD_z_int_values_all.append(df_y['SMD_z_exp'].values)


#%% Plot maps (individually)

# SMD plot UG100
i = 0
levels_   = range(25,41,1)
plt.figure(figsize=figsize_maps)
plt.contourf(yy_values_all[i], zz_values_all[i], SMD_values_all[i], levels = levels_, cmap='binary')
plt.colorbar(format = '%d',ticks=[25,30,35,40],label=label_SMD)
contour = plt.contour(yy_values_all[i], zz_values_all[i], SMD_values_all[i], levels = levels_,
                              colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(-12, 12)#(plot_bounds[0])
plt.ylim(0, 20)#(plot_bounds[1])
plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.savefig(folder_manuscript+'map_SMD_UG100.pdf')
plt.show()
plt.close()

# Flux plot UG100
levels_ = np.linspace(0,5,11)
plt.figure(figsize=figsize_maps)
plt.contourf(yy_values_all[i], zz_values_all[i], flux_values_all[i], levels = levels_, cmap='binary')
plt.colorbar(format = '%.1f',ticks=levels_,label=label_flux)
contour = plt.contour(yy_values_all[i], zz_values_all[i], flux_values_all[i], levels = levels_, 
                              colors= 'k', linewidths = 2*FFIG)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
plt.xlabel('$y ~ [mm]$')
plt.ylabel('$z ~ [mm]$')
plt.tight_layout()
#plt.axis('off')
#plt.title('$Flux$')
plt.xticks([-10, -5, 0, 5, 10])
plt.xlim(-12, 12)#(plot_bounds[0])
plt.ylim(0, 20)#(plot_bounds[1])
plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.savefig(folder_manuscript+'map_flux_UG100.pdf')
plt.show()
plt.close()       

#%% Plot maps with subplots       




for i in range(len(cases)):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize_maps_subplots)
    # Flux map
    levels_flux = np.linspace(0,5,11)
    cf1 = ax1.contourf(yy_values_all[i], zz_values_all[i], flux_values_all[i], levels = levels_flux, cmap='binary')
    divider = make_axes_locatable(ax1)
    cax = divider.new_vertical(size = '5%', pad = pad_cmap_maps)
    fig.add_axes(cax)
    cb = fig.colorbar(cf1, cax=cax, format = r'$%.1f$', orientation="horizontal",
                 ticks=np.linspace(0,5,6))
    cb.set_label(label_flux, labelpad=pad_title_maps)
    ax1.contour(yy_values_all[i], zz_values_all[i], flux_values_all[i], levels = levels_, 
                                  colors= 'k', linewidths = 2*FFIG)
    ax1.set_xlabel(y_label_)
    ax1.set_ylabel(z_label_)
    ax1.set_xticks([-10, -5, 0, 5, 10])
    ax1.set_xlim(-12, 12)#(plot_bounds[0])
    ax1.set_yticks(range(0, 24, 4))
    ax1.set_ylim(0, 20)#(plot_bounds[1])
    # SMD map
    levels_SMD  = range(25,41,1)
    cf2 = ax2.contourf(yy_values_all[i], zz_values_all[i], SMD_values_all[i], levels = levels_SMD, cmap='binary')
    divider = make_axes_locatable(ax2)
    cax = divider.new_vertical(size = '5%', pad = pad_cmap_maps)
    fig.add_axes(cax)
    cb = fig.colorbar(cf2, cax=cax, format = r'$%d$', orientation="horizontal",
                 ticks=[25,30,35,40])
    cb.set_label(label_SMD, labelpad=pad_title_maps)
    #fig.colorbar(cf2, ax=ax2, format = r'$%d$',ticks=[25,30,35,40],label=label_SMD)
    ax2.contour(yy_values_all[i], zz_values_all[i], SMD_values_all[i], levels = levels_SMD, 
                                  colors= 'k', linewidths = 2*FFIG)
    ax2.set_xlabel(y_label_)
    #ax2.set_ylabel(z_label_)
    ax2.set_xticks([-10, -5, 0, 5, 10])
    ax2.set_xlim(-12, 12)#(plot_bounds[0])
    ax2.set_yticks(range(0, 24, 4))
    ax2.set_ylim(0, 20)#(plot_bounds[1])
    #ax2.colorbar(format = '%d',ticks=[25,30,35,40],label=label_SMD)
    plt.tight_layout()
    if i == 0:
        save_title = 'maps_UG100.pdf'
    if i == 1:
        save_title = 'maps_UG75.pdf'
    plt.savefig(folder_manuscript+save_title)
    plt.show()
    plt.close()




#%% Plot spatially integrated values


plt.rcParams['legend.fontsize'] = 40*FFIG #50*FFIG



# Integrated with y
fig = plt.figure(figsize=figsize_)
ax = fig.add_subplot(111)
lns1 = ax.plot(z_int_values_all[0],flux_z_int_values_all[0],'-ok',label=r'$\mathrm{UG100},~q_l$')
lns2 = ax.plot(z_int_values_all[1],flux_z_int_values_all[1],'--ok',label=r'$\mathrm{UG75},~q_l$')
ax2 = ax.twinx()
lns3 = ax2.plot(z_int_values_all[0],SMD_z_int_values_all[0],'-^b',label=r'$\mathrm{UG100},~SMD$')
lns4 = ax2.plot(z_int_values_all[1],SMD_z_int_values_all[1],'--^b',label=r'$\mathrm{UG75},~SMD$')
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left')
#ax.legend(loc='upper left')
ax.grid()
ax.set_xlabel(z_label_)
ax.set_xlim((0,20))
ax.set_xticks([0,5,10,15,20])
ax.set_ylabel(label_flux_aver)
ax2.set_ylabel(label_SMD_aver)
ax.set_ylim((0,5))
ax.set_yticks(np.linspace(0,ax.get_ylim()[1], int(ax.get_ylim()[1]/1)+1 ))
ax2.set_ylim((15,45))
ax2.set_yticks([15,20,25,30,35,40,45])
ax2.tick_params(axis='y', colors='blue',pad=20)
ax2.yaxis.label.set_color('blue')
plt.tight_layout()
#plt.savefig(folder_manuscript+'integrated_fluxes_along_y.pdf')
plt.show()
plt.close()






# Integrated with y (switched axis)
fig = plt.figure(figsize=(FFIG*24,FFIG*20))
ax = fig.add_subplot(111)
lns1 = ax.plot(flux_z_int_values_all[0],z_int_values_all[0],'-ok',label=r'$\mathrm{UG100},~q_l$')
lns2 = ax.plot(flux_z_int_values_all[1],z_int_values_all[1],'--ok',label=r'$\mathrm{UG75},~q_l$')
ax2 = ax.twiny()
lns3 = ax2.plot(SMD_z_int_values_all[0],z_int_values_all[0],'-^b',label=r'$\mathrm{UG100},~SMD$')
lns4 = ax2.plot(SMD_z_int_values_all[1],z_int_values_all[1],'--^b',label=r'$\mathrm{UG75},~SMD$')
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left', ncol=2)
#ax.legend(loc='upper left')
ax.grid()
ax.set_ylabel(z_label_)
ax.set_ylim((0,20))
ax.set_yticks([0,5,10,15,20])
ax.set_xlabel(label_flux_aver)
ax2.set_xlabel(label_SMD_aver)
ax.set_xlim((0,5))
ax.set_xticks(np.linspace(0,ax.get_xlim()[1], int(ax.get_xlim()[1]/1)+1 ))
ax2.set_xlim((15,45))
ax2.set_xticks([15,20,25,30,35,40,45])
ax2.tick_params(axis='x', colors='blue',pad=20)
ax2.xaxis.label.set_color('blue')
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_fluxes_along_y.pdf')
plt.show()
plt.close()



#%%


# Integrated with z
fig = plt.figure(figsize=figsize_)
ax = fig.add_subplot(111)
ax.plot(y_int_values_all[0],flux_y_int_values_all[0],'-ok')
ax.plot(y_int_values_all[1],flux_y_int_values_all[1],'--ok')
ax2 = ax.twinx()
ax2.plot(y_int_values_all[0],SMD_y_int_values_all[0],'-^b')
ax2.plot(y_int_values_all[1],SMD_y_int_values_all[1],'--^b')
#ax.legend(loc='upper left')
ax.grid()
ax.set_xlabel(y_label_)
ax.set_xticks([-10,-5,0,5,10])
ax.set_xlim((-11,11))
ax.set_ylabel(label_flux_aver)
ax2.set_ylabel(label_SMD_aver)
ax.set_ylim((0,5))
ax.set_yticks(np.linspace(0,ax.get_ylim()[1], int(ax.get_ylim()[1]/1)+1 ))
ax2.set_ylim((20,37.5))
ax2.set_yticks([20,25,30,35])
ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
plt.tight_layout()
plt.savefig(folder_manuscript+'integrated_fluxes_along_z.pdf')
plt.show()
plt.close()

#%% Get SMDs integrated with profiles

SMD_int_y_cases = []; SMD_int_z_cases = []
for i in range(len(cases)):
    y_values = y_int_values_all[i]
    flux_along_y = flux_y_int_values_all[i]
    SMD_along_y  = SMD_y_int_values_all[i]
    SMD_int_y = get_SMD_from_integrated_profile(y_values, SMD_along_y, flux_along_y)
    
    z_values = z_int_values_all[i]
    flux_along_z   = flux_z_int_values_all[i]
    SMD_along_z  = SMD_z_int_values_all[i]
    SMD_int_z = get_SMD_from_integrated_profile(z_values, SMD_along_z, flux_along_z)
    
    SMD_int_y_cases.append(SMD_int_y)
    SMD_int_z_cases.append(SMD_int_z)

#%% Print SMDs
    
print('------ SMDs ------')
for i in range(len(cases)):
    print(f'  Case {labels_cases[i]}')
    print(f'      Arithmetic: {SMD_global_cases[i]}')
    print(f'   Flux-weighted: {SMD_global_flux_weighted_cases[i]}')
    print(f'    Int. along y: {SMD_int_y_cases[i]}')
    print(f'    Int. along z: {SMD_int_z_cases[i]}')

