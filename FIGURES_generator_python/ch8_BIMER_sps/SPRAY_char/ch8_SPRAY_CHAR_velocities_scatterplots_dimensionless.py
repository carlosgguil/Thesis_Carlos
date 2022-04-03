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
sys.path.append('../..')
from sli_functions import load_all_BIMER_global_sprays




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


folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/velocities/'

#%% Load sprays


# Parameters of simulations
params_simulation = {'RHO_L': 750, 'MU_L': 1.36e-3, 'U_L'  : 2.6,
                     'RHO_G': 0.82, 'MU_G': 2.39e-5, 'U_G'  : 56,
                     'SIGMA': 25e-3,
                     'D_inj': 0.3e-3}
    

# Load sprays
sp1, sp2, sp3 = load_all_BIMER_global_sprays(params_simulation)

sprays_list_all = [sp1, sp2, sp3]


#%% Parameters


format_separating_line = 'k'
linewidth_separating_line = 15*FFIG
linewidth_Ql = 6*FFIG





#x_label = r'$\mathrm{SMD}~[\mu \mathrm{m}]$' 
#y_label_ux = r'$u_x~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uy = r'$u_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uz = r'$u_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
x_label = r'$D~[\mu \mathrm{m}]$' 
y_label_ux = r'$u_c/u_g$'
y_label_uy = r'$v_c/u_l$'
y_label_uz = r'$w_c/u_l$'

label_legend_scatter   = r'$\mathrm{Droplets}$'
label_legend_u_mean    = r'$\mathrm{Arithmetic~mean}$'
label_legend_u_mean_vw = r'$\mathrm{VW~mean}$'

marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'k'
line_umean_vw_format = '--k' 
x_lim_ = [0,75]


#%% Plots 
# Choose a spray
s = 0 # DX
p = 2 # xD




spray = sprays_list_all[s][p]

u = spray.ucx/params_simulation['U_G']
u_mean = spray.uc_mean[0]/params_simulation['U_G']
u_VW   = spray.uc_mean_volume_weighted[0]/params_simulation['U_G']
v = spray.ucy/params_simulation['U_L']
v_mean = spray.uc_mean[1]/params_simulation['U_L']
v_VW   = spray.uc_mean_volume_weighted[1]/params_simulation['U_L']
w = spray.ucz/params_simulation['U_L']
w_mean = spray.uc_mean[2]/params_simulation['U_L']
w_VW   = spray.uc_mean_volume_weighted[2]/params_simulation['U_L']

# scatterplot u x
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.ux, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(spray.diam.values, u, c = spray.x, facecolors='none', s=marker_size_, label = label_legend_scatter, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[u_mean]*2,line_umean_format, label=label_legend_u_mean)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[u_VW]*2,line_umean_vw_format, label=label_legend_u_mean_vw)
plt.legend(loc='best',ncol=1)
plt.xlabel(x_label)
plt.ylabel(y_label_ux, labelpad=30*FFIG)
plt.xlim(x_lim_)
plt.ylim(0,1)
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
plt.scatter(spray.diam.values, v, c = spray.x, facecolors='none', s=marker_size_, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[v_mean]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[v_VW]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uy, labelpad=-40*FFIG)
plt.xlim(x_lim_)
plt.ylim((-10,10))
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
plt.scatter(spray.diam.values, w, c = spray.x, facecolors='none', s=marker_size_, cmap=cm.seismic) 
plt.plot([min(spray.diam.values),max(spray.diam.values)],[w_mean]*2,line_umean_format)
plt.plot([min(spray.diam.values),max(spray.diam.values)],[w_VW]*2,line_umean_vw_format)
plt.xlabel(x_label)
plt.ylabel(y_label_uz, labelpad=-40*FFIG)
plt.xlim(x_lim_)
plt.ylim((-10,10))
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
cbar.set_ticks(np.linspace(0,10,6))
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