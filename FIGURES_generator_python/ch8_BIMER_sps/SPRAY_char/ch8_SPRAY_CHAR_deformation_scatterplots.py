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
figsize_2 = (FFIG*18,FFIG*16)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part3_applications/figures_ch8_resolved/SPRAY_characterization/deformation/'

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


# axis labels
x_label_  = r'$x_c/d_\mathrm{inj}$' #r'$t~[\mathrm{ms}]$'
y_label_alpha_mean = r'$\overline{\alpha}_\mathrm{VW}$'
y_label_alpha_rms  = r'$\alpha_\mathrm{RMS}$'
y_label_beta_mean = r'$\overline{\beta}_\mathrm{VW}$'
y_label_beta_rms = r'$\beta_\mathrm{RMS}$'
y_label_alpha_beta_mean = r'$\overline{\alpha}_\mathrm{VW}, \overline{\beta}_\mathrm{VW}$'


# legend labels
label_DX15  = r'$\mathrm{DX}15$'
label_DX10  = r'$\mathrm{DX}10$'
label_DX07 = r'$\mathrm{DX}07$'
labels_OP = [label_DX07, label_DX10 , label_DX15]



label_xD03p33 = r'$x_c/d_\mathrm{inj} = 3.33$'
label_xD05p00 = r'$x_c/d_\mathrm{inj} = 5.00$'
label_xD06p67 = r'$x_c/d_\mathrm{inj} = 6.67$'
labels_ = [label_xD03p33, label_xD05p00, label_xD06p67]
# x coordinates
xD = [3.33,5,6.67]






y_label_alpha = r'$\alpha$'
y_label_beta = r'$\beta$'



marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'k'
line_umean_vw_format = '--k' 
x_lim_ = [0,250]

alpha_lim = (1,7)
beta_lim = (0,1)
alpha_ticks = ([1,3,5,7])

diam_min = 25
diam_max = 100

# Choose plane:
i_plane = 2

# filter outsiders
beta_max = 0.95
alpha_min = 1.05

# sphericity criteria
alpha_th = 2
beta_th  = 0.5

#%% Plots DX10

# Choose a spray
s = 1 # spray DX15

spray = sprays_list_all[s][i_plane]

alpha = []; beta = []; diameters = []
for i in range(spray.n_droplets):
    alpha_i = spray.alpha.values[i]
    beta_i = spray.beta.values[i]
    if alpha_i < alpha_min or beta_i > beta_max:
        continue
    else:
        alpha.append(alpha_i)
        beta.append(beta_i)
        diameters.append(spray.diam.values[i])


# scatterplot alpha-beta coloured by diameter
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(alpha, beta, c = diameters, facecolors='none', s=diameters, cmap=cm.seismic,
            vmin = diam_min, vmax = diam_max)
plt.xlabel(y_label_alpha)
plt.ylabel(y_label_beta, labelpad=20*FFIG)
plt.xlim(alpha_lim)
plt.xticks(alpha_ticks)
plt.ylim(beta_lim)
plt.title(labels_OP[s])
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'scatter_alpha_beta_DX10.png')
plt.show()
plt.close()

#%% Plots DX15
# Choose a spray
s = 2 # spray DX15

spray = sprays_list_all[s][i_plane]

alpha = []; beta = []; diameters = []
for i in range(spray.n_droplets):
    alpha_i = spray.alpha.values[i]
    beta_i = spray.beta.values[i]
    if alpha_i < alpha_min or beta_i > beta_max:
        continue
    else:
        alpha.append(alpha_i)
        beta.append(beta_i)
        diameters.append(spray.diam.values[i])
        


# scatterplot alpha-beta coloured by diameter
plt.figure(figsize=figsize_2)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.scatter(alpha, beta, c = diameters, facecolors='none', s=diameters, cmap=cm.seismic,
            vmin = diam_min, vmax = diam_max)
plt.xlabel(y_label_alpha)
#plt.ylabel(y_label_beta, labelpad=20*FFIG)
plt.xlim(alpha_lim)
plt.xticks(alpha_ticks)
plt.ylim(beta_lim)
plt.title(labels_OP[s])
plt.grid()
ax = plt.gca()
ax.yaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig(folder_manuscript+'scatter_alpha_beta_DX15.png')
plt.show()
plt.close()

#%% Standalone colormap diam

plt.rcParams['text.usetex'] = True

a = np.array([[diam_min,diam_max]])
plt.figure(figsize=(FFIG*1.5, FFIG*18))
#plt.figure(figsize=(fPic*1.5, fPic*18))
img = plt.imshow(a, cmap="seismic")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation="vertical", cax=cax)
cbar.set_label(r'$D~[\mu\mathrm{m}]$',labelpad=30)
cbar.set_ticks(np.linspace(diam_min,diam_max,4))
#plt.tight_layout()
plt.savefig(folder_manuscript+'scatterplots_colorbar_D.png',bbox_inches="tight")


#%% Calculate spray sphericity for all sprays

print(' % of spherical droplets (all sprays)')
print(' ----------------- ')
for m in range(len(sprays_list_all)):
    for n in range(len(sprays_list_all[m])):
        spray = sprays_list_all[m][n]
        count_total = 0
        count_spherical = 0
        for i in range(spray.n_droplets):
            alpha_i = spray.alpha.values[i]
            beta_i = spray.beta.values[i]
            if alpha_i < alpha_min or beta_i > beta_max:
                continue
            else:
                count_total += 1
                if alpha_i < alpha_th and beta_i > beta_th:
                    count_spherical += 1
        perc_sph =  count_spherical/count_total*100
        
        print(f'  Case {labels_OP[m]}, {spray.name}: {perc_sph:.5f}')
            
