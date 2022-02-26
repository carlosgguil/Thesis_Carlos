# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sprPost_calculations import get_sprays_list, get_discrete_spray
from sli_functions import load_all_SPS_global_sprays

FFIG = 0.5
plt.rcParams['xtick.labelsize'] = 60*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.markersize'] = 40*FFIG

figsize_ = (FFIG*24,FFIG*16)
figsize_maps = (FFIG*15,FFIG*15)
folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/'


#%% Format 

y_label_   = r'$y~[\mathrm{mm}]$'
z_label_   = r'$z~[\mathrm{mm}]$'

#%% Load sprays

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





# General keywords
parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/SPS_sprays"
filename   = "vol_dist_coarse"

# JICF SPS solutions 
filename        = "vol_dist_coarse"
sampling_planes_DX10 = ['x = 05 mm']


dirs_UG100_DX10 = ["q6uG100/sps_dx=10m"]
dirs_UG100_DX10 = [parent_dir +'/'+ d for d in dirs_UG100_DX10]

# Load spray
sprays_list_UG100_DX10 = get_sprays_list(True, sampling_planes_DX10, 
                                         dirs_UG100_DX10, 
                                         filename,
                                         params_simulation_UG100,
                                         CASE = 'JICF',
                                         sols_dirs_name = ".")


# Load grid
grids_lists = get_discrete_spray(True, sprays_list_UG100_DX10, [8]*2,
                                    None, params_simulation_UG100, 
                                    DIR = dirs_UG100_DX10[0],
                                    ADD_U_REL_TO_GRID = True)
grid = grids_lists[0][0]



#%% Calculate errors
N_EL_Z = grid.grid_size[1]
N_EL_Y = grid.grid_size[0]

du_rel_arr = [ [ [] for i in range(N_EL_Y) ] for j in range(N_EL_Z) ]
du_ul_arr  = [ [ [] for i in range(N_EL_Y) ] for j in range(N_EL_Z) ]
eps_arr    = [ [ [] for i in range(N_EL_Y) ] for j in range(N_EL_Z) ]
eps_ul_arr = [ [ [] for i in range(N_EL_Y) ] for j in range(N_EL_Z) ]
for m in range(N_EL_Z):
    for n in range(N_EL_Y):
        if not grid.SPRAYS_array[m][n].IS_EMPTY:
            
            u_rel_sps_norm_mn = np.linalg.norm(grid.SPRAYS_array[m][n].u_rel_SPS)
            u_rel_lgs_norm_mn = np.linalg.norm(grid.SPRAYS_array[m][n].u_rel_LGS)
            du_rel_mn = u_rel_lgs_norm_mn - u_rel_sps_norm_mn
            eps_mn    = du_rel_mn/u_rel_sps_norm_mn*100
            '''
            u_rel_sps_norm_mn = grid.SPRAYS_array[m][n].u_rel_SPS
            u_rel_lgs_norm_mn = grid.SPRAYS_array[m][n].u_rel_LGS
            du_rel_mn = np.linalg.norm(u_rel_lgs_norm_mn - u_rel_sps_norm_mn)
            eps_mn    = du_rel_mn/np.linalg.norm(u_rel_sps_norm_mn)*100
            #eps_mn = np.abs(eps_mn)
            '''
            u_l_sps_norm_mn = np.linalg.norm(grid.SPRAYS_array[m][n].u_mean)
            u_l_lgs_norm_mn = np.linalg.norm(grid.SPRAYS_array[m][n].u_mean_LGS)
            du_ul_mn = u_l_lgs_norm_mn - u_l_sps_norm_mn
            eps_ul_mn = du_ul_mn/u_l_sps_norm_mn*100
            '''
            u_l_sps_norm_mn = grid.SPRAYS_array[m][n].u_mean
            u_l_lgs_norm_mn = grid.SPRAYS_array[m][n].u_mean_LGS
            du_ul_mn = np.linalg.norm(u_l_lgs_norm_mn - u_l_sps_norm_mn)
            eps_ul_mn = du_ul_mn/np.linalg.norm(u_l_sps_norm_mn)*100
            '''
        else:
            eps_mn = np.nan
            du_rel_mn = np.nan
            du_ul_mn = np.nan
            eps_ul_mn = np.nan
            
        eps_arr[m][n] = eps_mn
        du_rel_arr[m][n] = du_rel_mn
        eps_ul_arr[m][n] = eps_ul_mn
        du_ul_arr[m][n] = du_ul_mn
    
eps_arr = np.array(eps_arr)      
map_eps =  np.ma.masked_where(np.isnan(eps_arr), eps_arr)

du_rel_arr = np.array(du_rel_arr)      
map_du_rel =  np.ma.masked_where(np.isnan(du_rel_arr), du_rel_arr)

eps_ul_arr = np.array(eps_ul_arr)
map_eps_ul = np.ma.masked_where(np.isnan(eps_ul_arr), eps_ul_arr)

du_ul_arr = np.array(du_ul_arr)      
map_du_ul =  np.ma.masked_where(np.isnan(du_ul_arr), du_ul_arr)

#%% Map u_mean_l

N_LEVELS  = 10
map_values = grid.map_ux_mean
max_level = np.nanmax(map_values.data)
min_level = np.nanmin(map_values.data)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]


plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_l~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Map SMD

N_LEVELS  = 10
map_values = grid.map_SMD
max_level = np.nanmax(map_values.data)
min_level = np.nanmin(map_values.data)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]


plt.figure(figsize=figsize_maps)
plt.title(r'$SMD~[\mu m]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Map vol flux

N_LEVELS  = 10
map_values = grid.map_vol_flux*1e2
max_level = np.nanmax(map_values.data)
min_level = np.nanmin(map_values.data)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]


plt.figure(figsize=figsize_maps)
plt.title(r'$q_l~[cm^3 s^{-1} cm^{-2}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Map uz

N_LEVELS  = 10
map_values = grid.map_uz_mean
max_level = np.nanmax(map_values.data)
min_level = np.nanmin(map_values.data)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]


plt.figure(figsize=figsize_maps)
plt.title(r'$w~[m s^{-1} ]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()


#%% Map u_mean_g SPS and LGS

max_level_SPS = np.nanmax(grid.map_ug_sps_mean_x.data); min_level_SPS = np.nanmin(grid.map_ug_sps_mean_x.data)
max_level_LGS = np.nanmax(grid.map_ug_lgs_mean_x.data); min_level_LGS = np.nanmin(grid.map_ug_lgs_mean_x.data)
max_level = max(max_level_SPS, max_level_LGS)
min_level = min(min_level_SPS, min_level_LGS)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]

# SPS
map_values = grid.map_ug_sps_mean_x
plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_{g,\mathrm{SPS}}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()


# LGS
map_values = grid.map_ug_lgs_mean_x
plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_{g,\mathrm{LGS}}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Map u_mean_LGS

map_values = grid.map_u_mean_lgs_x
max_level = np.nanmax(map_values.data)
min_level = np.nanmin(map_values.data)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]

plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_{l,\mathrm{LGS}}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Map u_rel SPS and LGS

max_level_SPS = np.nanmax(grid.map_u_rel_sps_x.data); min_level_SPS = np.nanmin(grid.map_u_rel_sps_x.data)
max_level_LGS = np.nanmax(grid.map_u_rel_lgs_x.data); min_level_LGS = np.nanmin(grid.map_u_rel_lgs_x.data)
max_level = max(max_level_SPS, max_level_LGS)
min_level = min(min_level_SPS, min_level_LGS)
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]

# SPS
map_values = grid.map_u_rel_sps_x
plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_\mathrm{rel,SPS}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()


# LGS
map_values = grid.map_u_rel_lgs_x
plt.figure(figsize=figsize_maps)
plt.title(r'$\overline{u}_\mathrm{rel,LGS}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%.1f',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

#%% Error maps

N_LEVELS = 10
min_level = -20
max_level = 20
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]

map_values = map_du_rel
plt.figure(figsize=figsize_maps)
plt.title(r'$\Delta u_\mathrm{rel}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%d',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

map_values = map_du_ul
plt.figure(figsize=figsize_maps)
plt.title(r'$\Delta u_\mathrm{l}~[\mathrm{m~s}^{-1}]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_)
plt.colorbar(format = '%d',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()






N_LEVELS = 10
min_level = -80
max_level = 100
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]

map_values = map_eps
plt.figure(figsize=figsize_maps)
plt.title(r'$\varepsilon_{u_\mathrm{rel}}~[\%]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_,extend='max')
plt.colorbar(format = '%d',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()

N_LEVELS = 10
min_level = -40
max_level = 40
levels_ = [max_level*i/(N_LEVELS-1) + min_level*(1-i/(N_LEVELS-1)) for i in range(N_LEVELS)]


map_values = map_eps_ul
plt.figure(figsize=figsize_maps)
plt.title(r'$\varepsilon_{u_\mathrm{l}}~[\%]$')
plt.contourf(grid.yy_center, grid.zz_center, map_values, cmap='binary',
             levels = levels_,extend='max')
plt.colorbar(format = '%d',ticks=levels_)
contour = plt.contour(grid.yy_center, grid.zz_center, map_values, 
                      levels = levels_, colors= 'k', linewidths = 2*FFIG)
plt.xlabel(y_label_)
plt.ylabel(z_label_)
plt.tight_layout()
#plt.title('$SMD ~ [\mu m]$')
#plt.xticks([-10, -5, 0, 5, 10])
#plt.xlim(-12, 12)#(plot_bounds[0])
#plt.ylim(0, 20)#(plot_bounds[1])
#plt.yticks(range(0, 20, 4))
plt.tight_layout()
plt.show()
plt.close()