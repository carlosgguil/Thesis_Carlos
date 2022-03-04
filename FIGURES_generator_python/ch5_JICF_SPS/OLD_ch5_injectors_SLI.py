# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
FFIG = 0.5
figsize_ = (FFIG*22,FFIG*13)
import matplotlib.pyplot as plt
import numpy as np

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/injectors_SLI/'

plt.rcParams['xtick.labelsize'] = 70*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 70*FFIG#40*FFIG
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
cmap_      = 'binary' #'binary, 'Greys'
N_LEVELS = 10
format_ = '%d'
xlabel_ = 'y [mm]'
ylabel_ = 'z [mm]'

#%% Case and grid

case = 'uG100_dx20_x15'
parent_grid = grids_list[0][0]


# Aspect ratio
dy = np.diff(parent_grid.bounds[0])[0]
dz = np.diff(parent_grid.bounds[1])[0]
AR = dy/dz 


#%% Plot u mean


map_values = parent_grid.map_ux_mean
fig_title  = '$\overline{u}$ at '# +plane_name
bar_label  = '$\overline{u}$ [m s$^{-1}$]'


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
plt.savefig(folder_manuscript+case+'_ux_mean_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% Plot u RMS


map_values = parent_grid.map_ux_rms
fig_title  = '$u_\mathrm{RMS}$ at '# +plane_name
bar_label  = '$u_\mathrm{RMS}$ [m s$^{-1}$]'


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
plt.savefig(folder_manuscript+case+'_ux_RMS_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()


#%% Plot v mean


map_values = parent_grid.map_uy_mean
fig_title  = '$\overline{v}$ at '# +plane_name
bar_label  = '$\overline{v}$ [m s$^{-1}$]'


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
plt.savefig(folder_manuscript+case+'_uy_mean_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% Plot v RMS


map_values = parent_grid.map_uy_rms
fig_title  = '$v_\mathrm{RMS}$ at '# +plane_name
bar_label  = '$v_\mathrm{RMS}$ [m s$^{-1}$]'



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
plt.savefig(folder_manuscript+case+'_uy_RMS_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()



#%% Plot w mean


map_values = parent_grid.map_uz_mean
fig_title  = '$\overline{w}$ at '# +plane_name
bar_label  = '$\overline{w}$ [m s$^{-1}$]'


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
plt.savefig(folder_manuscript+case+'_uz_mean_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()

#%% Plot w RMS


map_values = parent_grid.map_uz_rms
fig_title  = '$w_\mathrm{RMS}$ at '# +plane_name
bar_label  = '$w_\mathrm{RMS}$ [m s$^{-1}$]'


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
plt.savefig(folder_manuscript+case+'_uz_RMS_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()




#%% Plot SMD

map_values = parent_grid.map_SMD
fig_title = 'SMD at ' #+plane_name
bar_label  = '$SMD$ [$\mu \mathrm{m}$]'


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
plt.savefig(folder_manuscript+case+'_SMD_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()



#%% Plot Volume flux

map_values = parent_grid.map_vol_flux*1e2
fig_title  = 'Volume flux at '# +plane_name
bar_label  = 'Volume flux [cm$^3$ s$^{-1}$ cm$^{-2}$]'
format_ = '%.1f'


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
plt.savefig(folder_manuscript+case+'_volume_flux_map.eps',format='eps',dpi=1000)
plt.show()
plt.close()


