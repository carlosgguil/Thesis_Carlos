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
figsize_2 = (FFIG*18,FFIG*16)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/SPRAY_characterization/deformation/'

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

labels_OP = [r'UG75_DX10', r'UG75_DX20', r'UG100_DX10', r'UG100_DX20', r'UG100_DX20_NT']


#x_label = r'$\mathrm{SMD}~[\mu \mathrm{m}]$' 
#y_label_ux = r'$u_x~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uy = r'$u_y~[\mathrm{m}~\mathrm{s}^{-1}]$'
#y_label_uz = r'$u_z~[\mathrm{m}~\mathrm{s}^{-1}]$'
x_label = r'$D~[\mu \mathrm{m}]$' 
y_label_alpha = r'$\alpha$'
y_label_beta = r'$\beta$'

label_legend_scatter   = r'$\mathrm{Droplets}$'
label_legend_u_mean    = r'$\mathrm{Arithmetic~mean}$'
label_legend_u_mean_vw = r'$\mathrm{VW~mean}$'

marker_size_   = 200*FFIG
color_markers_ = 'black'
line_umean_format = 'k'
line_umean_vw_format = '--k' 
x_lim_ = [0,250]

alpha_lim = (1,9)
beta_lim = (0,1)
alpha_ticks = ([1,3,5,7,9,])

diam_min = 10
diam_max = 630

# Choose plane:
i_plane = 1 # 0: x = 5 mm ; 1: x = 10 mm; 2: x = 15 mm

# filter outsiders
beta_max = 0.99
alpha_min =  1.01

# sphericity criteria
alpha_th = 2
beta_th  = 0.5

#%% Plots UG100_DX10

# Choose a spray
s = 2 # spray UG100_DX10

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

alpha_UG100_DX10 = alpha
beta_UG100_DX10  = beta

# scatterplot alpha-beta coloured by diameter
plt.figure(figsize=figsize_)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.plot([1,alpha_th],[beta_th]*2,'k',linewidth=15*FFIG)
plt.plot([alpha_th]*2,[beta_th,1],'k',linewidth=15*FFIG)
plt.scatter(alpha, beta, c = diameters, facecolors='none', s=diameters, cmap=cm.seismic,
            vmin = diam_min, vmax = diam_max)
plt.xlabel(y_label_alpha)
plt.ylabel(y_label_beta, labelpad=20*FFIG)
plt.xlim(alpha_lim)
plt.xticks(alpha_ticks)
plt.ylim(beta_lim)
plt.title(label_UG100_DX10)
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'scatter_alpha_beta_UG10_DX10.png')
plt.show()
plt.close()

#%% Plots UG100_DX20
# Choose a spray
s = 3 # spray UG100_DX10

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
        
alpha_UG100_DX20 = alpha
beta_UG100_DX20  = beta

# scatterplot alpha-beta coloured by diameter
plt.figure(figsize=figsize_2)
#plt.scatter(spray.diam.values, spray.uy, facecolors='none', s=marker_size_, color=color_markers_) 
plt.plot([1,alpha_th],[beta_th]*2,'k',linewidth=15*FFIG)
plt.plot([alpha_th]*2,[beta_th,1],'k',linewidth=15*FFIG)
plt.scatter(alpha, beta, c = diameters, facecolors='none', s=diameters, cmap=cm.seismic,
            vmin = diam_min, vmax = diam_max)
plt.xlabel(y_label_alpha)
#plt.ylabel(y_label_beta, labelpad=20*FFIG)
plt.xlim(alpha_lim)
plt.xticks(alpha_ticks)
plt.ylim(beta_lim)
plt.title(label_UG100_DX20)
plt.grid()
ax = plt.gca()
ax.yaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig(folder_manuscript+'scatter_alpha_beta_UG10_DX20.png')
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
cbar.set_ticks(np.linspace(diam_min,diam_max,5))
#plt.tight_layout()
plt.savefig(folder_manuscript+'scatterplots_colorbar_D.png',bbox_inches="tight")

#%% Calculate spray sphericity based on Zuzio (2018) and Herrmann (2010) criteria
# for two sprays

# UG100_DX10
count = 0
for i in range(len(alpha_UG100_DX10)):
    alpha_i = alpha_UG100_DX10[i]
    beta_i  = beta_UG100_DX10[i]
    
    if alpha_i < alpha_th and beta_i > beta_th:
        count += 1
    
perc_spherical_droplets_UG100_DX10 = count/len(alpha_UG100_DX10)*100

# UG100_DX20
count = 0
for i in range(len(alpha_UG100_DX20)):
    alpha_i = alpha_UG100_DX20[i]
    beta_i  = beta_UG100_DX20[i]
    
    if alpha_i < alpha_th and beta_i > beta_th:
        count += 1
    
perc_spherical_droplets_UG100_DX20 = count/len(alpha_UG100_DX20)*100


print(' % of spherical droplets (plotted sprays')
print(f'  UG100_DX10: {perc_spherical_droplets_UG100_DX10:.5f}')
print(f'  UG100_DX20: {perc_spherical_droplets_UG100_DX20:.5f}')
print(' ----------------- ')

#%% Calculate spray sphericity for all sprays

print(' % of spherical droplets (all sprays')
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
            
