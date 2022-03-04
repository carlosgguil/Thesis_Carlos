"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

   



from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays




FFIG = 0.5
SCALE_FACTOR = 1e9
PLOT_ADAPTATION_ITERS = True
# rcParams for plots
plt.rcParams['xtick.labelsize'] = 90*FFIG # 80*FFIG 
plt.rcParams['ytick.labelsize'] = 90*FFIG # 80*FFIG
plt.rcParams['axes.labelsize']  = 90*FFIG #80*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 90*FFIG #80*FFIG
plt.rcParams['legend.fontsize'] = 60*FFIG #50*FFIG
plt.rcParams['font.size'] = 50*FFIG
plt.rcParams['lines.linewidth'] =  8*FFIG #6*FFIG
plt.rcParams['lines.markersize'] =  40*FFIG #6*FFIG
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc']      = 'upper right'
plt.rcParams['text.usetex'] = True
#figsize_ = (FFIG*22,FFIG*16)
figsize_ = (FFIG*24,FFIG*24)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/velocities/'

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

# Load sprays
sp1, sp2, sp3, sp4, sp5 = load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100)

sprays_list_all = [sp1, sp2, sp3, sp4, sp5]

#%% Parameters


format_separating_line = 'k'
linewidth_separating_line = 15*FFIG
linewidth_Ql = 6*FFIG


#x_label = r'$\mathrm{SMD}~[\mu \mathrm{m}]$' 
#y_label_ux = r'$u_x~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uy = r'$u_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uz = r'$u_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
x_label = r'$D~[\mu \mathrm{m}]$' 
y_label_ux = r'$u~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy = r'$v~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz = r'$w~[\mathrm{m}~\mathrm{s}^{-1}]$'


labelpad_ = -130
label_y = r'$y~[\mathrm{mm}]$'
label_z = r'$z~[\mathrm{mm}]$'

label_legend_scatter   = r'$\mathrm{Droplets}$'
label_legend_u_mean    = r'$\mathrm{Arithmetic~mean}$'
label_legend_u_mean_vw = r'$\mathrm{VW~mean}$'

marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'b'
line_umean_vw_format = '--b' 
x_lim_ = [0,600]


#%% Plots 
# Choose a spray
s = 2 # UG100_DX10


'''
# scatterplot 2 x planes together

spray_x05 = sprays_list_all[s][0]
spray_x10 = sprays_list_all[s][1]
plt.figure(figsize=figsize_)
#plt.title(r'$u_g = 100~\mathrm{m}~\mathrm{s}^{-1}$',pad=40*FFIG)
plt.scatter(spray_x05.diam.values, spray_x05.ux, s=marker_size_, color='black') 
plt.scatter(spray_x10.diam.values, spray_x10.ux, facecolors='none', s=marker_size_, color='blue') 
plt.legend(loc='best',ncol=1)
plt.xlabel()
plt.tight_layout(pad=0)
#plt.savefig(folder_manuscript+'ug100_uz_rms.pdf')
plt.show()
plt.close()
'''

spray = sprays_list_all[s][1]


#%% 
a = np.array([[0,float(round(max(spray.z)))]])
z_coord_ticks = np.linspace(a[0][0],a[0][1],5)



# scatterplot u x
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.ux, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.ux, c = spray.z, facecolors='none', s=marker_size_, label = label_legend_scatter, cmap=cm.inferno) 
plt.scatter(-10,-10, c=-1, s=marker_size_)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[0]]*2,line_umean_format, label=label_legend_u_mean)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[0]]*2,line_umean_vw_format, label=label_legend_u_mean_vw)
plt.legend(loc='best',ncol=1)
plt.xlabel(x_label)
plt.ylabel(y_label_ux, labelpad=0*FFIG)
plt.xlim(x_lim_)
plt.ylim(0,110)
ax = plt.gca()
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('top', size='5%',pad='4%')
colorbar(ax.get_children()[0], cax=cax, orientation='horizontal', 
         ticks=z_coord_ticks)
cax.xaxis.set_ticks_position('top')
cax.set_xlabel(label_z,labelpad=labelpad_)
#cbar = plt.colorbar(orientation="horizontal")
#cbar.set_label(label_z,labelpad=labelpad_)
#cbar.set_ticks(z_coord_ticks)
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_ux_D.pdf')
plt.savefig(folder_manuscript+'scatter_ux_D.png')
plt.show()
plt.close()

#%%

y_coord_ticks = np.linspace(-8,8,5)


# scatterplot u y
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.uy, c = spray.y, facecolors='none', s=marker_size_, cmap=cm.inferno) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[1]]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[1]]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uy, labelpad=-50*FFIG)
plt.xlim(x_lim_)
ax = plt.gca()
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('top', size='5%',pad='4%')
colorbar(ax.get_children()[0], cax=cax, orientation='horizontal', 
         ticks=y_coord_ticks)
cax.xaxis.set_ticks_position('top')
cax.set_xlabel(label_y,labelpad=labelpad_)
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_uy_D.pdf')
plt.savefig(folder_manuscript+'scatter_uy_D.png')
plt.show()
plt.close()


#%% scatterplot u z


a = np.array([[0,float(round(max(spray.z)))]])
z_coord_ticks = np.linspace(a[0][0],a[0][1],5)


plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uz, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.uz, c = spray.z, facecolors='none', s=marker_size_, cmap=cm.inferno) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[2]]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[2]]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uz, labelpad=-50*FFIG)
plt.xlim(x_lim_)
ax = plt.gca()
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('top', size='5%',pad='4%')
colorbar(ax.get_children()[0], cax=cax, orientation='horizontal', 
         ticks=z_coord_ticks)
cax.xaxis.set_ticks_position('top')
cax.set_xlabel(label_z,labelpad=labelpad_)
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_uz_D.pdf')
plt.savefig(folder_manuscript+'scatter_uz_D.png')
plt.show()
plt.close()




