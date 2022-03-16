# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:22:35 2021

@author: d601630
"""


FFIG = 0.5
figsize_ = (FFIG*22,FFIG*13)
figsize_histo = (FFIG*18,FFIG*13)
figsize_scatt = (FFIG*18,FFIG*13)


from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays, load_all_SPS_grids, plot_grid, plot_grid_highlight_probe

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/injectors_SLI_extra/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/'

plt.rcParams['xtick.labelsize'] = 80*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 80*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 70*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 70*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'



# Default values
cbar_ticks = None
u_limits = None
u_rms_limits = None
ql_limits = None
plot_limits = None
N_LEVELS = 10
format_ = '%d'
xlabel_ = 'y [mm]'
ylabel_ = 'z [mm]'


# histogram parameters
bars_width = 1
x_label_pad = 4*FFIG


# scatterplots parameters

marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'k'
line_umean_vw_format = '--k' 

labels_u = [r'$u~[\mathrm{m}~\mathrm{s}^{-1}]$',
            r'$v~[\mathrm{m}~\mathrm{s}^{-1}]$',
            r'$w~[\mathrm{m}~\mathrm{s}^{-1}]$']
SMD_label = r'$D~[\mu \mathrm{m}]$' 


# spray and maps stuff
# tags for saving
cases = [['uG75_dx10_x05', 'uG75_dx10_x10'],
         ['uG75_dx20_x05', 'uG75_dx20_x10', 'uG75_dx20_x15'],
         ['uG100_dx10_x05', 'uG100_dx10_x10'],
         ['uG100_dx20_x05', 'uG100_dx20_x10', 'uG100_dx20_x15'],
         ['uG100_dx20_x05_NT', 'uG100_dx20_x10_NT']]

# plot limits
plot_bounds = [[(-10,10),(0,12)],
               [(-11,11),(0,12)],
               [(-5,5),(0,8)],
               [(-11,11),(0,12)],
               [(-10,10),(0,9)]]

PLOT_QL_MIN = False

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
sp1, sp2, sp3, sp4, sp5 = load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100)
sprays_list = [sp1, sp2, sp3, sp4, sp5]

# load grids
grids_list = load_all_SPS_grids(sprays_list)


#%% choose spray
i = 2
j = 0

# save tags
case = cases[i][j]

# plot limits
plot_limits = plot_bounds[i]

# grid
parent_grid = grids_list[i][j]

# mean velocity limits
u_limits = (29.5, 80)

#%% choose elements to highlight
m_ql_max = 3
n_ql_max = 3
color_ql_max = 'blue'

spray_ql_max = parent_grid.SPRAYS_array[n_ql_max][m_ql_max]


m_ql_min = 2
n_ql_min = 1
color_ql_min = 'red'

spray_ql_min = parent_grid.SPRAYS_array[n_ql_min][m_ql_min]


#%% A plotear muchacho

        



#dy = np.diff(parent_grid.bounds[0])[0]
#dz = np.diff(parent_grid.bounds[1])[0]
dy = np.diff(plot_limits[1])[0]
dz = np.diff(plot_limits[0])[0]
AR = dz/dy

cmap_      = 'binary' #'binary, 'Greys'


#%% Plot Volume flux

map_values = parent_grid.map_vol_flux*1e2
fig_title  = 'Volume flux at '# +plane_name
bar_label  = 'Volume flux [cm$^3$ s$^{-1}$ cm$^{-2}$]'
format_ = '%.1f'


plt.figure(figsize=(AR*FFIG*15.2,FFIG*15))
if ql_limits:
    min_level = ql_limits[0]
    max_level = ql_limits[1]
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
plot_grid_highlight_probe(parent_grid, m_ql_max, n_ql_max, color=color_ql_max)
if PLOT_QL_MIN:
    plot_grid_highlight_probe(parent_grid, m_ql_min, n_ql_min, color=color_ql_min)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_volume_flux_map.pdf' )
plt.show()
plt.close()
        
#%% Plot u mean
        
        
map_values = parent_grid.map_ux_mean
fig_title  = '$\overline{u}$ at '# +plane_name
bar_label  = '$\overline{u}$ [m s$^{-1}$]'

        
plt.figure(figsize=(AR*FFIG*15,FFIG*15))
if u_limits:
    min_level = u_limits[0]
    max_level = u_limits[1]
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
plot_grid_highlight_probe(parent_grid, m_ql_max, n_ql_max, color=color_ql_max)
if PLOT_QL_MIN:
    plot_grid_highlight_probe(parent_grid, m_ql_min, n_ql_min, color=color_ql_min)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_ux_mean_map.pdf')
plt.show()
plt.close()


#%% Plot u_VW mean
        
        
map_values = parent_grid.map_ux_mean_vw
fig_title  = '$\overline{u}_\mathrm{VW}$ at '# +plane_name
bar_label  = '$\overline{u}_\mathrm{VW}$ [m s$^{-1}$]'


plt.figure(figsize=(AR*FFIG*15,FFIG*15))
if u_limits:
    min_level = u_limits[0]
    max_level = u_limits[1]
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
plot_grid_highlight_probe(parent_grid, m_ql_max, n_ql_max, color=color_ql_max)
if PLOT_QL_MIN:
    plot_grid_highlight_probe(parent_grid, m_ql_min, n_ql_min, color=color_ql_min)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_ux_mean_VW_map.pdf')
plt.show()
plt.close()

#%% Plot u RMS


map_values = parent_grid.map_ux_rms
fig_title  = '$u_\mathrm{RMS}$ at '# +plane_name
bar_label  = '$u_\mathrm{RMS}$ [m s$^{-1}$]'


plt.figure(figsize=(AR*FFIG*15,FFIG*15))
if u_rms_limits:
    min_level = u_rms_limits[0]
    max_level = u_rms_limits[1]
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
plot_grid_highlight_probe(parent_grid, m_ql_max, n_ql_max, color=color_ql_max)
if PLOT_QL_MIN:
    plot_grid_highlight_probe(parent_grid, m_ql_min, n_ql_min, color=color_ql_min)
plt.tight_layout()
plt.savefig(folder_manuscript+case+'_ux_RMS_map.pdf')
plt.show()
plt.close()
        

#%% Plot histogram and scatterplots

spray = spray_ql_max
i = 0 # velocity component




data = spray.u_distr[i].values
n, bins       = np.histogram(np.sort(data), spray.n_bins)

# mean velocities for hist
vel_mean = spray.u_mean[i]
vel_min_CI = spray.u_mean[i] + spray.u_rms[i]
vel_max_CI = spray.u_mean[i] - spray.u_rms[i]

# VW velocities for hist
vel_mean_VW = spray.u_mean_volume_weighted[i]
vel_min_VW_CI = spray.u_mean_volume_weighted[i] + spray.u_rms_volume_weighted[i]
vel_max_VW_CI = spray.u_mean_volume_weighted[i] - spray.u_rms_volume_weighted[i]



plt.figure(figsize=figsize_histo) 
ax = plt.gca()
plt.hist(np.sort(data), spray.n_bins, color='black', rwidth = bars_width*0.8, 
         density = True, label=r'$\mathrm{PDF}$',zorder=250)

# mean velocities lines
plt.plot([vel_mean]*2,[0,1],'k',label=r'$\overline{u}$')
#plt.plot([vel_min_CI]*2,[0,1],':k')
#plt.plot([vel_max_CI]*2,[0,1],':k')
# mean VW velocities lines
plt.plot([vel_mean_VW]*2,[0,1],'--k',label=r'$\overline{u}_\mathrm{VW}$')
#plt.plot([vel_min_VW_CI]*2,[0,1],':b')
#plt.plot([vel_max_VW_CI]*2,[0,1],':b')
plt.ylim(0,0.04)
plt.yticks([0,0.01,0.02,0.03,0.04])
plt.xlabel(labels_u[i], labelpad = x_label_pad)
plt.ylabel(r'$\mathrm{PDF}$')
#plt.yticks(y_ticks_all[c])
#ax.yaxis.set_ticklabels([])
plt.grid(axis='y', alpha=0.75)
#plt.title(titles_xplanes[i])
plt.tight_layout()
plt.legend(loc='best')
plt.savefig(folder_manuscript+case+'_vel_histogram.pdf')
plt.show()
plt.close()



#%% Histogram of SMD

data = spray.diam.values
n, bins       = np.histogram(np.sort(data), spray.n_bins)

D_bins = bins
D_bins[-1] = D_bins[-1]*1.001
D_counts = n

plt.figure(figsize=figsize_histo) 
ax = plt.gca()
plt.hist(np.sort(data), spray.n_bins, color='black', rwidth = bars_width*0.8, 
         density = True, label=r'$\mathrm{PDF}$',zorder=250)

# SMD
plt.plot([spray.SMD]*2,[0,1],'k',label=r'$\mathrm{SMD}$')
#plt.plot([vel_min_VW_CI]*2,[0,1],':b')
#plt.plot([vel_max_VW_CI]*2,[0,1],':b')
plt.ylim(1e-5,5e-2)
plt.xlabel(SMD_label, labelpad = x_label_pad)
plt.xlim()
plt.ylabel(r'$\mathrm{PDF}$')
plt.yscale('log')
plt.yticks([1e-5,1e-4,1e-3,1e-2])
#plt.yticks(y_ticks_all[c])
#ax.yaxis.set_ticklabels([])
plt.grid(axis='y', alpha=0.75)
#plt.title(titles_xplanes[i])
plt.tight_layout()
plt.legend(loc='best')
plt.savefig(folder_manuscript+case+'_size_histogram.pdf')
plt.show()
plt.close()

#%% scatterplot 

# get info for sectional approach
u_bins = np.zeros(len(D_counts))
for n in range(spray.n_droplets):
    D_n = spray.diam.values[n]
    u_n = spray.u_distr[i].values[n]
    
    for j in range(len(D_bins)-1):
        D_bin_low = D_bins[j]
        D_bin_upp = D_bins[j+1]
        
        if (D_n >= D_bin_low) and (D_n < D_bin_upp):
            u_bins[j] += u_n
       
u_bins = np.nan_to_num(u_bins/D_counts)
u_bins = np.insert(u_bins, 0, u_bins[0], axis=0)
    

    



plt.figure(figsize=figsize_scatt)
#plt.scatter(spray.diam.values, spray.uz, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.u_distr[i].values, facecolors='none', edgecolor='b',s=marker_size_,label=r'$\mathrm{Droplets}$') 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[i]]*2,line_umean_format,label=r'$\overline{u}$')
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[i]]*2,line_umean_vw_format,label=r'$\overline{u}_\mathrm{VW}$')
plt.step(D_bins,u_bins,'r',linewidth=8*FFIG,label=r'$\overline{u}~\mathrm{per~section}$')
plt.plot([spray.SMD]*2,[0,1e4],'g',label=r'$\mathrm{SMD}$')
plt.xlabel(SMD_label)
plt.ylabel(labels_u[i])
#plt.xlim(x_lim_)
plt.ylim((20,100))
plt.yticks((20,40,60,80,100))
plt.grid()
plt.tight_layout()
plt.legend(loc='best')
plt.savefig(folder_manuscript+case+'_vel_scatter.pdf')
#plt.savefig(folder_manuscript+'scatter_uz_D.pdf')
#plt.savefig(folder_manuscript+'scatter_uz_D.png')
plt.show()
plt.close()
    