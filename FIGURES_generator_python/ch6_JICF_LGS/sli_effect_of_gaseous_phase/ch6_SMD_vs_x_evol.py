
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""





import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

import pickle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import numpy as np
import pandas as pd
from functions_expe_comparison import average_along_y, average_along_z
from functions_expe_comparison import get_SMD_from_integrated_profile, get_SMD_flux_weighted
from sprPost_calculations import get_discrete_spray, get_sprays_list
from sprPost_functions import get_grids_common_boundaries
import sprPost_plot as sprPlot


# Change size of figures 
FFIG = 0.5
#mpl.rcParams['font.size'] = 40*fPic
plt.rcParams['xtick.labelsize']  = 50*FFIG
plt.rcParams['ytick.labelsize']  = 50*FFIG
plt.rcParams['axes.labelsize']   = 50*FFIG
plt.rcParams['axes.labelpad']    = 30*FFIG
plt.rcParams['axes.titlesize']   = 50*FFIG
plt.rcParams['legend.fontsize']  = 40*FFIG
plt.rcParams['lines.linewidth']  = 7*FFIG
plt.rcParams['lines.markersize'] = 30*FFIG #20*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True
figsize_ = (FFIG*18,FFIG*13)
figsize_expe = (FFIG*21.7,FFIG*12.4)
figsize_along_z  = (FFIG*10,FFIG*15)

label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD = r'$SMD$ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'




SMD_lim_along_z = (10,40)
ql_lim_along_z = (0,2.5)
z_lim = (0,33)
SMD_lim_along_y = (15,35)
ql_lim_along_y = (0,2.5)
y_lim = (-12.5,12.5)

# for maps
flux_levels_ = np.linspace(0,5.5,12)
SMD_levels_ = np.linspace(15,40,10)

folder_manuscript='C:/Users/Carlos Garcia/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch6_lagrangian_JICF/params_gaseous_initial_conditions/'
folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/LGS_sprays_LAST'


store_variables_folder = 'store_variables' # 'store_variables' 'prev_store_variables'

folder_APTE = folder + '/apte_model_calibration_u_vw_lognorm/k1_0p05_k2_1p0/store_variables/'
folder_full_no_ALM  = folder + '/FULL_domain/LGS_no_ALM/'+store_variables_folder+'/'
folder_full_ALM_initial = folder + '/FULL_domain/LGS_ALM_initial/'+store_variables_folder+'/'
folder_full_FDC_0p10 = folder + '/FULL_domain/LGS_ALM_FDC_0p10/'+store_variables_folder+'/'
folder_full_FDC_0p24 = folder + '/FULL_domain/LGS_ALM_FDC_0p24/'+store_variables_folder+'/'
folder_full_FDC_0p30 = folder + '/FULL_domain/LGS_ALM_FDC_0p30/'+store_variables_folder+'/'


label_APTE = r'$\mathrm{Prescribed}$'
label_full_no_ALM = r'$\mathrm{No~ALM}$'
label_full_ALM_initial = r'$\mathrm{ALM~initial}$'
label_full_ALM_FDC_0p10 = r'$\mathrm{ALM~tilted}$'
label_full_ALM_FDC_0p24 = r'$\mathrm{ALM~optimal}$'
label_full_ALM_FDC_0p30 = r'$\mathrm{ALM~forced}$'



formats = {'APTE':'-k', 
           'full_no_ALM':':k', 
           'full_ALM_initial':'-b',
           'full_ALM_FDC_0p10':'-r',
           'full_ALM_FDC_0p24':'--y',
           'full_ALM_FDC_0p30':'-g'}

SMD_to_read = 'SMD_FW' # 'SMD', 'SMD_FW'

#SPS
format_SPS = '--^k'
label_SPS = r'$\mathrm{UG}100\_\mathrm{DX}10$'
SMD_from_SPS = 80.2

#SPS
format_SPS = '--^k'
label_SPS = r'$\mathrm{UG}100\_\mathrm{DX}10$'
SMD_from_SPS = 80.2

x_SPS     = [5, 10]
SMD_SPS_x = [SMD_from_SPS, 79.9]

#%% Experimental data and simulation parameters (do not touch)
folder_expe = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/DLR_data/'
data_int_y_exp = pd.read_csv(folder_expe + '1210_01_data_integrated_y_exp.csv')
data_int_z_exp = pd.read_csv(folder_expe + '1210_01_data_integrated_z_exp.csv')

z_int_exp  = data_int_y_exp['z_values']
flux_z_exp = data_int_y_exp['flux_z_exp']
SMD_z_exp  = data_int_y_exp['SMD_z_exp']

y_int_exp  = data_int_z_exp['y_values']
flux_y_exp = data_int_z_exp['flux_y_exp']
SMD_y_exp  = data_int_z_exp['SMD_y_exp']


# experimental SMD
SMD_expe = 31

# estimate expe errors
#error_SMD  = 0.26
#error_flux = 0.37
error_SMD  = 0.14
error_flux = 0.2


error_q_y_expe = flux_y_exp*error_flux
error_q_z_expe = flux_z_exp*error_flux
error_SMD_y_expe = SMD_y_exp*error_SMD
error_SMD_z_expe = SMD_z_exp*error_SMD


params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']


# expe data for maps
with open(folder_expe+'/pickle_map_expe_high_we', 'rb') as f:
    obj = pickle.load(f)
    expe_y_values = obj[0]
    expe_z_values = obj[1]
    expe_yy_values = obj[2]
    expe_zz_values = obj[3]
    expe_flux_values = obj[4]
    expe_SMD_values = obj[5]



width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG




#%% get data


df_SMD_evol_APTE = pd.read_csv(folder_APTE+'SMD_evolution.csv')
df_SMD_evol_full_no_ALM  = pd.read_csv(folder_full_no_ALM+'SMD_evolution.csv')
df_SMD_evol_full_ALM_initial = pd.read_csv(folder_full_ALM_initial+'SMD_evolution.csv')
df_SMD_evol_full_ALM_FDC_0p10 = pd.read_csv(folder_full_FDC_0p10+'SMD_evolution_modified.csv')
df_SMD_evol_full_ALM_FDC_0p24 = pd.read_csv(folder_full_FDC_0p24+'SMD_evolution.csv')
df_SMD_evol_full_ALM_FDC_0p30 = pd.read_csv(folder_full_FDC_0p30+'SMD_evolution.csv')

x_APTE = df_SMD_evol_APTE['x'].values
x_APTE = np.insert(x_APTE,0,5)
SMD_APTE = df_SMD_evol_APTE[SMD_to_read].values
SMD_APTE = np.insert(SMD_APTE,0,SMD_from_SPS)

x_full_no_ALM = df_SMD_evol_full_no_ALM['x'].values
x_full_no_ALM = np.insert(x_full_no_ALM,0,5)
SMD_full_no_ALM = df_SMD_evol_full_no_ALM[SMD_to_read].values
SMD_full_no_ALM = np.insert(SMD_full_no_ALM,0,SMD_from_SPS)

x_full_ALM_initial = df_SMD_evol_full_ALM_initial['x'].values
x_full_ALM_initial = np.insert(x_full_ALM_initial,0,5)
SMD_full_ALM_initial = df_SMD_evol_full_ALM_initial[SMD_to_read].values
SMD_full_ALM_initial = np.insert(SMD_full_ALM_initial,0,SMD_from_SPS)

x_full_ALM_FDC_0p10 = df_SMD_evol_full_ALM_FDC_0p10['x'].values
x_full_ALM_FDC_0p10= np.insert(x_full_ALM_FDC_0p10,0,5)
SMD_full_ALM_FDC_0p10 = df_SMD_evol_full_ALM_FDC_0p10[SMD_to_read].values
SMD_full_ALM_FDC_0p10 = np.insert(SMD_full_ALM_FDC_0p10,0,SMD_from_SPS)

x_full_ALM_FDC_0p24 = df_SMD_evol_full_ALM_FDC_0p24['x'].values
x_full_ALM_FDC_0p24= np.insert(x_full_ALM_FDC_0p24,0,5)
SMD_full_ALM_FDC_0p24 = df_SMD_evol_full_ALM_FDC_0p24[SMD_to_read].values
SMD_full_ALM_FDC_0p24 = np.insert(SMD_full_ALM_FDC_0p24,0,SMD_from_SPS)

x_full_ALM_FDC_0p30 = df_SMD_evol_full_ALM_FDC_0p30['x'].values
x_full_ALM_FDC_0p30= np.insert(x_full_ALM_FDC_0p30,0,5)
SMD_full_ALM_FDC_0p30 = df_SMD_evol_full_ALM_FDC_0p30[SMD_to_read].values
SMD_full_ALM_FDC_0p30 = np.insert(SMD_full_ALM_FDC_0p30,0,SMD_from_SPS)


# for plotting diameters
cases_all = ['prescribed',
             'full_no_ALM',
             'full_ALM_initial',
             'full_ALM_FDC_0p10',
             'full_ALM_FDC_0p24',
             'full_ALM_FDC_0p30',]
x_all = [x_APTE,
         x_full_no_ALM,
         x_full_ALM_initial,
         x_full_ALM_FDC_0p10,
         x_full_ALM_FDC_0p24,
         x_full_ALM_FDC_0p30]
SMDs_all = [SMD_APTE,
            SMD_full_no_ALM,
            SMD_full_ALM_initial,
            SMD_full_ALM_FDC_0p10,
            SMD_full_ALM_FDC_0p24,
            SMD_full_ALM_FDC_0p30]



#%% plot


figsize_SMD_evol_along_x = (FFIG*30,FFIG*12)


x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)

plt.figure(figsize=figsize_SMD_evol_along_x)
# Experimentql result
plt.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
plt.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
# LGS results
plt.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)
plt.plot(x_APTE,SMD_APTE, formats['APTE'], label=label_APTE)
plt.plot(x_full_no_ALM,SMD_full_no_ALM, formats['full_no_ALM'], label=label_full_no_ALM)
#plt.plot(x_full_ALM_initial,SMD_full_ALM_initial, formats['full_ALM_initial'], label=label_full_ALM_initial)
plt.plot(x_full_ALM_FDC_0p10,SMD_full_ALM_FDC_0p10, formats['full_ALM_FDC_0p10'], label=label_full_ALM_FDC_0p10)
plt.plot(x_full_ALM_FDC_0p24,SMD_full_ALM_FDC_0p24, formats['full_ALM_FDC_0p24'], label=label_full_ALM_FDC_0p24)
#plt.plot(x_full_ALM_FDC_0p30,SMD_full_ALM_FDC_0p30, formats['full_ALM_FDC_0p30'], label=label_full_ALM_FDC_0p30)
#plt.clabel(contour, colors='k', fmt='%d', fontsize=40)
#plt.xscale('log')
plt.xlabel(label_x) 
plt.ylabel(label_SMD) 
plt.legend(loc='best', ncol=1)
plt.grid()
plt.tight_layout()
#plt.axis('off')
#plt.title('$\mathrm{Jaegle~(2008)}$')
plt.xticks(x_ticks_SMD_evol)
plt.yticks(y_ticks_SMD_evol)
plt.xlim(4,81)#(plot_bounds[0])
#plt.xlim(13.5,16.5)
plt.ylim(00,82)#(plot_bounds[1])
#plt.savefig(folder_manuscript+'SMD_vs_x_gaseous_BCs_comparison.pdf')
#plt.show()
plt.close()      


#%%

x_ticks_SMD_evol = np.arange(5,85,5)
y_ticks_SMD_evol = np.arange(0,81,10)


fig, ax1 = plt.subplots(figsize=figsize_SMD_evol_along_x)

# data for main plot

# Experimentql result
ax1.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax1.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
ax1.plot(x_APTE,SMD_APTE, 
         formats['APTE'], label=label_APTE)

ax1.plot(x_full_no_ALM,SMD_full_no_ALM, 
         formats['full_no_ALM'], label=label_full_no_ALM)
ax1.plot(x_full_ALM_initial,SMD_full_ALM_initial, 
         formats['full_ALM_initial'], label=label_full_ALM_initial)
ax1.plot(x_full_ALM_FDC_0p10,SMD_full_ALM_FDC_0p10, 
         formats['full_ALM_FDC_0p10'], label=label_full_ALM_FDC_0p10)
ax1.plot(x_full_ALM_FDC_0p24,SMD_full_ALM_FDC_0p24, 
         formats['full_ALM_FDC_0p24'], label=label_full_ALM_FDC_0p24)
ax1.plot(x_full_ALM_FDC_0p30,SMD_full_ALM_FDC_0p30, 
         formats['full_ALM_FDC_0p30'], label=label_full_ALM_FDC_0p30)
ax1.plot(x_SPS, SMD_SPS_x, format_SPS, label=label_SPS)
'''
ax1.plot(x_full_ALM_FDC_0p10,SMD_full_ALM_FDC_0p10, 
         formats['full_ALM_FDC_0p10'], label=label_full_ALM_FDC_0p10)
ax1.plot(x_full_ALM_FDC_0p24,SMD_full_ALM_FDC_0p24, 
         formats['full_ALM_FDC_0p24'], label=label_full_ALM_FDC_0p24)
'''
# characteristics main plot
ax1.set_xlabel(label_x)
ax1.set_ylabel(label_SMD)
ax1.set_xlim(4,83)
ax1.set_ylim(00,85)
ax1.set_xticks(x_ticks_SMD_evol)
ax1.set_yticks(y_ticks_SMD_evol)
#ax1.legend(loc='best',ncol=2)
ax1.legend(bbox_to_anchor=(1.30,0.85))
ax1.grid()
#ax1.grid(which='major',linestyle='-',linewidth=4*FFIG)
#ax1.grid(which='minor',linestyle='--')

linewidth_cotas = 5*FFIG
linewidth_arrow = 5*FFIG
head_width_ = 2
head_length_ = 0.7
x_arrows = 20
y_arrows = 58
l_arrows = 4

ax1.plot([x_arrows]*2, [0,63], '--k', linewidth = linewidth_cotas)
ax1.arrow(x_arrows, y_arrows, l_arrows, 0, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
ax1.arrow(x_arrows, y_arrows, -l_arrows, 0, head_width=head_width_, head_length=head_length_, 
          linewidth=linewidth_arrow, color='k', shape = 'full', length_includes_head=True, clip_on = False)
ax1.text(x_arrows-l_arrows*3, y_arrows, 
         '$\mathrm{Exponential}$ \n $\mathrm{decay}$',fontsize=40*FFIG)
ax1.text(x_arrows+l_arrows*1.1, y_arrows, 
         '$\mathrm{Linear}$ \n $\mathrm{decay}$',fontsize=40*FFIG)
# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
#ip = InsetPosition(ax1, [0.45,0.40,0.3,0.3])
ip = InsetPosition(ax1, [0.55,0.40,0.3,0.5])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

# data for embedded plot
ax2.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
ax2.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)

ax2.plot(x_APTE,SMD_APTE, 
         formats['APTE'], label=label_APTE)
ax2.plot(x_full_no_ALM,SMD_full_no_ALM, 
         formats['full_no_ALM'], label=SMD_full_ALM_FDC_0p24)
ax2.plot(x_full_ALM_initial,SMD_full_ALM_initial, 
         formats['full_ALM_initial'], label=label_full_ALM_initial)
ax2.plot(x_full_ALM_FDC_0p10,SMD_full_ALM_FDC_0p10, 
         formats['full_ALM_FDC_0p10'], label=label_full_ALM_FDC_0p10)
ax2.plot(x_full_ALM_FDC_0p24,SMD_full_ALM_FDC_0p24, 
         formats['full_ALM_FDC_0p24'], label=label_full_ALM_FDC_0p24)
ax2.plot(x_full_ALM_FDC_0p30,SMD_full_ALM_FDC_0p30, 
         formats['full_ALM_FDC_0p30'], label=label_full_ALM_FDC_0p30)


# characteristics embedded plot
ax2.set_xlim((79,81))
ax2.set_ylim((9,36))
#ax2.set_ylim((liquid_volume_UG100_DX20[index_1],liquid_volume_UG100_DX20[index_2]))
labelsize_embedded_plot = 40*FFIG
ax2.xaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.yaxis.set_tick_params(labelsize=labelsize_embedded_plot)
ax2.grid(which='major',linestyle='-',linewidth=4*FFIG)
ax2.grid(which='minor',linestyle='--')


# draw rectangle
'''
w_rect = ax2.get_xlim()[1] - ax2.get_xlim()[0]+0.6
h_rect = ax2.get_ylim()[1] - ax2.get_ylim()[0]
rect = Rectangle((ax2.get_xlim()[0]-0.3,ax2.get_ylim()[0]),w_rect,h_rect, 
                 linewidth=1,edgecolor='k',facecolor='none',zorder = 2)
ax1.add_patch(rect)
'''

# Some ad hoc tweaks.
#ax1.set_ylim(y_lim_)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
plt.tight_layout()
plt.savefig(folder_manuscript+'SMD_vs_x_gaseous_BCs_comparison.pdf')
plt.show()
plt.close()


#%% Plot SMDs and deviations with experiments
x0_linear = 20 # start of linear decrease region
for i in range(len(cases_all)):
    case_i = cases_all[i]
    x_i   = x_all[i]
    SMD_i = SMDs_all[i]
    
    
    index_x0_linear = np.where(x_i == x0_linear)[0][0]
    SMD_x0_linear = SMD_i[index_x0_linear]
    SMD_x80 = SMD_i[-1]
    
    decay_rate = (x0_linear - SMD_x80)/(80 - x0_linear)
    
    eps_SMD = (SMD_x80 - SMD_expe)/SMD_expe*100
    print(f'  Case {case_i}: SMD_x20 = {SMD_x0_linear:.3f}, SMD_x80 = {SMD_x80:.3f}, error: {eps_SMD:.3f}, decay rate: {decay_rate:.3f}')
