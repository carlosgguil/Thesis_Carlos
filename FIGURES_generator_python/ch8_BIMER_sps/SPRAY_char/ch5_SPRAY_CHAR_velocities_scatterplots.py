"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

   



from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
figsize_ = (FFIG*22,FFIG*16)

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

# axis labels
x_label_  = r'$x~[\mathrm{mm}]$' #r'$t~[\mathrm{ms}]$'
y_label_ux_rms  = r'$u_{x,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy_mean = r'$\overline{u}_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz_mean = r'$\overline{u}_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy_rms = r'$u_{y,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz_rms = r'$u_{z,\mathrm{RMS}}~[\mathrm{m}~\mathrm{s}^{-1}]$'

# legend labels
label_UG75_DX10  = r'$\mathrm{UG}75\_\mathrm{DX}10$'
label_UG75_DX20  = r'$\mathrm{UG}75\_\mathrm{DX}20$'
label_UG100_DX10 = r'$\mathrm{UG}100\_\mathrm{DX}10$'
label_UG100_DX20 = r'$\mathrm{UG}100\_\mathrm{DX}20$'
label_UG100_DX20_NT = r'$\mathrm{UG100}\_\mathrm{DX20}\_\mathrm{NT}$'
labels_OP = [label_UG75_DX10 , label_UG75_DX20,
                label_UG100_DX10, label_UG100_DX20,
                label_UG100_DX20_NT]


#x_label = r'$\mathrm{SMD}~[\mu \mathrm{m}]$' 
#y_label_ux = r'$u_x~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uy = r'$u_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uz = r'$u_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
x_label = r'$D~[\mu \mathrm{m}]$' 
y_label_ux = r'$u~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uy = r'$v~[\mathrm{m}~\mathrm{s}^{-1}]$'
y_label_uz = r'$w~[\mathrm{m}~\mathrm{s}^{-1}]$'

label_legend_scatter   = r'$\mathrm{Droplets}$'
label_legend_u_mean    = r'$\mathrm{Arithmetic~mean}$'
label_legend_u_mean_vw = r'$\mathrm{VW~mean}$'

marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'k'
line_umean_vw_format = '--k' 
x_lim_ = [0,250]


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

# scatterplot u x
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.ux, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.ux, c = spray.z, facecolors='none', s=marker_size_, label = label_legend_scatter, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[0]]*2,line_umean_format, label=label_legend_u_mean)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[0]]*2,line_umean_vw_format, label=label_legend_u_mean_vw)
plt.legend(loc='best',ncol=1)
plt.xlabel(x_label)
plt.ylabel(y_label_ux, labelpad=0*FFIG)
plt.xlim(x_lim_)
plt.ylim(0,120)
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_ux_D.pdf')
plt.savefig(folder_manuscript+'scatter_ux_D.png')
plt.show()
plt.close()

#%%
# scatterplot u y
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.uy, c = spray.y, facecolors='none', s=marker_size_, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[1]]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[1]]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uy, labelpad=-40*FFIG)
plt.xlim(x_lim_)
plt.ylim((-80,80))
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_uy_D.pdf')
plt.savefig(folder_manuscript+'scatter_uy_D.png')
plt.show()
plt.close()
#%%

# scatterplot u z
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uz, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, spray.uz, c = spray.z, facecolors='none', s=marker_size_, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean[2]]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[spray.u_mean_volume_weighted[2]]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uz, labelpad=-40*FFIG)
plt.xlim(x_lim_)
plt.ylim((-90,90))
plt.grid()
plt.tight_layout()
#plt.savefig(folder_manuscript+'scatter_uz_D.pdf')
plt.savefig(folder_manuscript+'scatter_uz_D.png')
plt.show()
plt.close()

#%% Standalone colormap z
#import pylab as pl
plt.rcParams['text.usetex'] = True

a = np.array([[0,float(round(max(spray.z)))]])
plt.figure(figsize=(FFIG*18, FFIG*1.5))
#plt.figure(figsize=(fPic*1.5, fPic*18))
img = plt.imshow(a, cmap="seismic")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation="horizontal", cax=cax)
cbar.set_label(r'$z~[\mathrm{mm}]$',labelpad=-130)
cbar.set_ticks(np.linspace(a[0][0],a[0][1],5))
#plt.tight_layout()
plt.savefig(folder_manuscript+'scatterplots_colorbar_z.png',bbox_inches="tight")


#%% Standalone colormap y

plt.rcParams['text.usetex'] = True

a = np.array([[-8,8]])
plt.figure(figsize=(FFIG*18, FFIG*1.5))
#plt.figure(figsize=(fPic*1.5, fPic*18))
img = plt.imshow(a, cmap="seismic")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation="horizontal", cax=cax)
cbar.set_label(r'$y~[\mathrm{mm}]$',labelpad=-130)
cbar.set_ticks(np.linspace(a[0][0],a[0][1],5))
#plt.tight_layout()
plt.savefig(folder_manuscript+'scatterplots_colorbar_y.png',bbox_inches="tight")