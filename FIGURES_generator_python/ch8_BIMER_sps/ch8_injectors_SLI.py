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
from sli_functions import load_all_BIMER_global_sprays, load_all_BIMER_grids, plot_grid

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/injectors_SLI/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/'

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



# Default values
cbar_ticks = None
variable_limits = None
plot_limits = None
N_LEVELS = 10
format_ = '%d'
xlabel_ = 'y$_c$ [mm]'
ylabel_ = 'z$_c$ [mm]'


# tags for saving
cases = [['dx10_xD05p00', 'dx10_xD06p67'],
         ['dx15_xD05p00', 'dx15_xD06p67']]

# plot limits
plot_bounds = [[(-2.5,2.5),(0,3)],
               [(-2,2),(0,3)]]

#%% Load sprays and grids


# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    

# Load sprays
_, spDX10, spDX15 = load_all_BIMER_global_sprays(params_simulation)

spDX10 = spDX10[1:]
spDX15 = spDX15[1:]


sprays_list_all = [spDX10, spDX15]



# load grids
grids_list = load_all_BIMER_grids(sprays_list_all)



#%% A plotear muchacho

for i in range(len(grids_list)):
    # bounds
    plot_limits = plot_bounds[i]
    for j in range(len(grids_list[i])):
        
        #if i != 4:# or j != 0:
        #    continue
        
        
        # save tags
        case = cases[i][j]
        # grid
        parent_grid = grids_list[i][j]
        
        
        #dy = np.diff(parent_grid.bounds[0])[0]
        #dz = np.diff(parent_grid.bounds[1])[0]
        dy = np.diff(plot_limits[1])[0]
        dz = np.diff(plot_limits[0])[0]
        AR = dz/dy
        
        cmap_      = 'binary' #'binary, 'Greys'
        
        #%% Plot u mean
        
        
        map_values = parent_grid.map_ux_mean
        fig_title  = '$\overline{u}_c$ at '# +plane_name
        bar_label  = '$\overline{u}_c$ [m s$^{-1}$]'
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_ux_mean_map.pdf')
        plt.show()
        plt.close()
        
        #%% Plot u RMS
        
        
        map_values = parent_grid.map_ux_rms
        fig_title  = '$u_{c,\mathrm{RMS}}$ at '# +plane_name
        bar_label  = '$u_{c,\mathrm{RMS}}$ [m s$^{-1}$]'
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_ux_RMS_map.pdf')
        plt.show()
        plt.close()
        
        
        #%% Plot v mean
        
        
        map_values = parent_grid.map_uy_mean
        fig_title  = '$\overline{v}_c$ at '# +plane_name
        bar_label  = '$\overline{v}_c$ [m s$^{-1}$]'
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_uy_mean_map.pdf' )
        plt.show()
        plt.close()
        
        #%% Plot v RMS
        
        
        map_values = parent_grid.map_uy_rms
        fig_title  = '$v_{c,\mathrm{RMS}}$ at '# +plane_name
        bar_label  = '$v_{c,\mathrm{RMS}}$ [m s$^{-1}$]'
        
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_uy_RMS_map.pdf' )
        plt.show()
        plt.close()
        
        
        
        #%% Plot w mean
        
        
        map_values = parent_grid.map_uz_mean
        fig_title  = '$\overline{w}_c$ at '# +plane_name
        bar_label  = '$\overline{w}_c$ [m s$^{-1}$]'
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_uz_mean_map.pdf' )
        plt.show()
        plt.close()
        
        #%% Plot w RMS
        
        
        map_values = parent_grid.map_uz_rms
        fig_title  = '$w_{c,\mathrm{RMS}}$ at '# +plane_name
        bar_label  = '$w_{c,\mathrm{RMS}}$ [m s$^{-1}$]'
        
        
        plt.figure(figsize=(AR*FFIG*15,FFIG*15))
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_uz_RMS_map.pdf' )
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
        
        levels_map = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]
        contour = plt.contour(parent_grid.yy_center, parent_grid.zz_center, map_values, 
                              levels = levels_map, colors= 'k', linewidths = 2*FFIG)
        plt.contourf(parent_grid.yy_center, parent_grid.zz_center, map_values,
                     levels = levels_map, cmap = cmap_ , extend=extend_)
        cbar = plt.colorbar(format=format_)
        cbar.set_label(bar_label)
        #plt.title(title)
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
        cbar.set_label(bar_label)
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plot_grid(parent_grid, ADD_TO_FIGURE = True)
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_volume_flux_map.pdf' )
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
        #plt.title(title)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        if plot_limits:
            plt.xlim(plot_limits[0][0], plot_limits[0][1])
            plt.ylim(plot_limits[1][0], plot_limits[1][1])
        plot_grid(parent_grid, ADD_TO_FIGURE = True)
        plt.tight_layout()
        plt.savefig(folder_manuscript+case+'_convergence_map.pdf' )
        plt.show()
        plt.close()
        

