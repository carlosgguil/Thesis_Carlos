# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""


FFIG = 0.5
figsize_ = (FFIG*22,FFIG*13)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays, load_all_SPS_grids, plot_grid
from sprPost_functions import get_grids_common_boundaries, pickle_load

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/injectors_SLI/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/SPS_sprays/q6uG100/'

plt.rcParams['xtick.labelsize'] = 80*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 70*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 70*FFIG
plt.rcParams['legend.fontsize'] = 70*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# Default values
cbar_ticks = None
variable_limits = None
plot_limits = None
N_LEVELS = 10
format_ = '%d'
xlabel_ = 'y [mm]'
ylabel_ = 'z [mm]'


# tags for saving
cases = ['uG100_dx10_x05', 'uG100_dx10_x05_quadtrees']

# plot limits
plot_bounds = [[(-5,5),(0,8.2)]]

#%% Load sprays and grids


# Parameters of simulations
params_simulation_UG100 = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                           'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                           'SIGMA': 22e-3,
                           'D_inj': 0.45e-3}

params_simulation_UG75 = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 17.5,
                          'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 75,
                          'SIGMA': 22e-3,
                          'D_inj': 0.45e-3}
params_simulation_UG100['Q_inj'] = np.pi/4*params_simulation_UG100['D_inj']**2*params_simulation_UG100['U_L']
params_simulation_UG75['Q_inj'] = np.pi/4*params_simulation_UG75['D_inj']**2*params_simulation_UG75['U_L']


# load sprays
spray_UG100_DX10 = pickle_load(folder+'sps_dx=10m/store_variables_SLI_manuscript/sprays_list_x=05mm')
grid_UG100_DX10  = pickle_load(folder+'sps_dx=10m/store_variables_SLI_manuscript/grids_list_x=05mm')

grid_UG100_DX10_quadtrees   = pickle_load(folder+'sps_dx=10m/store_variables_quadtrees/grids_list_x=05mm')

'''
sp1, sp2, sp3, sp4, sp5 = load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100,
                                                     save_dir = 'store_variables_SLI_manuscript')
sprays_list = [sp1, sp2, sp3, sp4, sp5]

# load grids
grids_list = load_all_SPS_grids(sprays_list, save_dir = 'store_variables_SLI_manuscript')
'''
grids_list = [grid_UG100_DX10, grid_UG100_DX10_quadtrees]

pad_title_maps = -120

#%% A plotear muchacho

for i in range(len(grids_list)):
    
    plot_limits = plot_bounds[0]
    parent_grid = grids_list[i]
    case = cases[i]
    
    
    dy = np.diff(plot_limits[1])[0]
    dz = np.diff(plot_limits[0])[0]
    AR = dz/dy
    
    cmap_      = 'binary' #'binary, 'Greys'

        
#%% Plot u mean and u_vw
        
        
    map_values_arith = parent_grid.map_ux_mean
    map_values_vw = parent_grid.map_ux_mean_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    
    # quadtrees low limit
    if case == 'uG100_dx10_x05_quadtrees':
        max_level = 82
    

    
    # arithmetic 
    map_values = parent_grid.map_ux_mean   
    fig_title  = '$\overline{u}$ at '# +plane_name
    bar_label  = '$\overline{u}$ [m s$^{-1}$]'
    
    plt.figure(figsize=(1.02*AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_, 
                        orientation="vertical")
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_ux_mean_map.pdf')
    plt.show()
    plt.close()
    
    
    # VW
    map_values = parent_grid.map_ux_mean_vw
    fig_title  = '$u_\mathrm{VW}$ at '# +plane_name
    bar_label  = '$u_\mathrm{VW}$ [m s$^{-1}$]'
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_, 
                        orientation="vertical")
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_ux_mean_vw_map.pdf')
    plt.show()
    plt.close()
    
    #%% Plot u RMS
    
    
    map_values_arith = parent_grid.map_ux_rms
    map_values_vw = parent_grid.map_ux_rms_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    
    # arithmetic 
    map_values = parent_grid.map_ux_rms
    fig_title  = '$u_\mathrm{RMS}$ at '# +plane_name
    bar_label  = '$u_\mathrm{RMS}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_ux_RMS_map.pdf')
    plt.show()
    plt.close()
    
    
    # VW 
    map_values = parent_grid.map_ux_rms
    fig_title  = '$u_\mathrm{RMS,VW}$ at '# +plane_name
    bar_label  = '$u_\mathrm{RMS,VW}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_ux_RMS_vw_map.pdf')
    plt.show()
    plt.close()
        
    #%% Plot v mean
    
            
    map_values_arith = parent_grid.map_uy_mean
    map_values_vw = parent_grid.map_uy_mean_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    # Arithmetic
    map_values = parent_grid.map_uy_mean
    fig_title  = '$\overline{v}$ at '# +plane_name
    bar_label  = '$\overline{v}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uy_mean_map.pdf' )
    plt.show()
    plt.close()
    
    
    # VW
    map_values = parent_grid.map_uy_mean_vw
    fig_title  = '$v_\mathrm{VW}$ at '# +plane_name
    bar_label  = '$v_\mathrm{VW}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uy_mean_vw_map.pdf' )
    plt.show()
    plt.close()
    
    #%% Plot v RMS
    
    map_values_arith = parent_grid.map_uy_rms
    map_values_vw = parent_grid.map_uy_rms_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    # Arithmetic
    map_values = parent_grid.map_uy_rms
    fig_title  = '$v_\mathrm{RMS}$ at '# +plane_name
    bar_label  = '$v_\mathrm{RMS}$ [m s$^{-1}$]'
    
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
   
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uy_RMS_map.pdf' )
    plt.show()
    plt.close()
    
    
    # VW
    map_values = parent_grid.map_uy_rms_vw
    fig_title  = '$v_\mathrm{RMS,VW}$ at '# +plane_name
    bar_label  = '$v_\mathrm{RMS,VW}$ [m s$^{-1}$]'
    
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
   
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uy_RMS_vw_map.pdf' )
    plt.show()
    plt.close()
    
    
    
    #%% Plot w mean
    
            
    map_values_arith = parent_grid.map_uz_mean
    map_values_vw = parent_grid.map_uz_mean_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    # Arithmetic
    map_values = parent_grid.map_uz_mean
    fig_title  = '$\overline{w}$ at '# +plane_name
    bar_label  = '$\overline{w}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uz_mean_map.pdf' )
    plt.show()
    plt.close()
    
    
    # VW
    map_values = parent_grid.map_uz_mean_vw
    fig_title  = '$w_\mathrm{VW}$ at '# +plane_name
    bar_label  = '$w_\mathrm{VW}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uz_mean_vw_map.pdf' )
    plt.show()
    plt.close()
    
    
    #%% Plot w RMS
    
    
    
    
            
    map_values_arith = parent_grid.map_uz_rms
    map_values_vw = parent_grid.map_uz_rms_vw
    
    min_level_arith = np.nanmin(map_values_arith.data)
    min_level_vw    = np.nanmin(map_values_vw.data)
    max_level_arith = np.nanmax(map_values_arith.data)
    max_level_vw    = np.nanmax(map_values_vw.data)
    extend_ = 'neither'
    
    min_level = min(min_level_vw,min_level_arith)
    max_level = max(max_level_vw,max_level_arith)
    
    
    # Arithmetic
    map_values = parent_grid.map_uz_rms
    fig_title  = '$w_\mathrm{RMS}$ at '# +plane_name
    bar_label  = '$w_\mathrm{RMS}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uz_RMS_map.pdf' )
    plt.show()
    plt.close()
    
    
    # Arithmetic
    map_values = parent_grid.map_uz_rms_vw
    fig_title  = '$w_\mathrm{RMS,VW}$ at '# +plane_name
    bar_label  = '$w_\mathrm{RMS,VW}$ [m s$^{-1}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15,FFIG*15))    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_uz_RMS_vw_map.pdf' )
    plt.show()
    plt.close()
            

        
        
    #%% Plot SMD
    
    map_values = parent_grid.map_SMD
    fig_title = 'SMD at ' #+plane_name
    bar_label  = '$SMD$ [$\mu \mathrm{m}$]'
    
    
    plt.figure(figsize=(AR*FFIG*15.2,FFIG*15))
    if variable_limits:
        min_level = variable_limits[0]
        max_level = variable_limits[1]
        condition_min_level = min_level > np.nanmin(map_values.data)
        condition_max_level = max_level < np.nanmax(map_values.data)
        extend_ = 'neither'
        if condition_min_level:
            extend_ = 'min'
        if condition_max_level:
            extend_ = 'max'
        if ((condition_min_level) and (condition_max_level)):
            extend_ = 'both'
    else:
        min_level = np.nanmin(map_values.data)
        max_level = np.nanmax(map_values.data)
        extend_ = 'neither'
    
    # quadtrees low limit
    if case == 'uG100_dx10_x05_quadtrees':
        min_level = 33
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_SMD_map.pdf' )
    plt.show()
    plt.close()
    
    
    
    #%% Plot Volume flux
    
    map_values = parent_grid.map_vol_flux*1e2
    fig_title  = 'Volume flux at '# +plane_name
    bar_label  = 'Volume flux [cm$^3$ s$^{-1}$ cm$^{-2}$]'
    format_ = '%.1f'
    
    
    plt.figure(figsize=(AR*FFIG*15.2,FFIG*15))
    if variable_limits:
        min_level = variable_limits[0]
        max_level = variable_limits[1]
        condition_min_level = min_level > np.nanmin(map_values.data)
        condition_max_level = max_level < np.nanmax(map_values.data)
        extend_ = 'neither'
        if condition_min_level:
            extend_ = 'min'
        if condition_max_level:
            extend_ = 'max'
        if ((condition_min_level) and (condition_max_level)):
            extend_ = 'both'
    else:
        min_level = np.nanmin(map_values.data)
        max_level = np.nanmax(map_values.data)
        extend_ = 'neither'
    
    levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
    contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                          levels = levels_map, colors= 'k', linewidths = 2*FFIG)
    plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                 levels = levels_map, cmap = cmap_ , extend=extend_)
    cbar = plt.colorbar(format=format_)
    #cbar.set_label(bar_label)
    plt.title(bar_label)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    if case == 'uG100_dx10_x05_quadtrees':
        plot_grid(parent_grid, ADD_TO_FIGURE = True,  PLOT_QUADTREES = True)
    else:
        plot_grid(parent_grid, ADD_TO_FIGURE = True)
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_volume_flux_map.pdf' , bbox_inches="tight")
    plt.show()
    plt.close()
    
    
    
    
         

    
    #%% Plot Convergence
    
    map_values = parent_grid.map_statsConv
    bar_label  = 'Not conv. ~~~~~~ Converged'
    format_ = '%d'
    
    plt.figure(figsize=(AR*FFIG*14.5,FFIG*15))   
    
    levels_map = [0,  1]
    cmap_ = plt.cm.get_cmap('viridis',2)
    plt.pcolor(parent_grid.yy_edges, parent_grid.zz_edges, map_values,
               vmin = levels_map[0], vmax = levels_map[-1], cmap = cmap_)
    cbar = plt.colorbar(format=format_)
    cbar.set_label(bar_label)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticks([])
    cbar.set_label(bar_label)
    #cbar.ax.set_xticklabels(['Converged', 'Not converged'])  # horizontal colorbar
    plt.title(r'$\mathrm{Convergence}$')
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if plot_limits:
        plt.xlim(plot_limits[0][0], plot_limits[0][1])
        plt.ylim(plot_limits[1][0], plot_limits[1][1])
    if case == 'uG100_dx10_x05_quadtrees':
        plot_grid(parent_grid, ADD_TO_FIGURE = True,  PLOT_QUADTREES = True)
    else:
        plot_grid(parent_grid, ADD_TO_FIGURE = True)
    plt.tight_layout()
    plt.savefig(folder_manuscript+case+'_convergence_map.pdf' )
    plt.show()
    plt.close()
    

