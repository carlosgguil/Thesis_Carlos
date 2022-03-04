
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""



import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')
sys.path.append('..')
from sli_functions import load_all_SPS_global_sprays, load_all_SPS_grids


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions_expe_comparison import average_along_y, average_along_z
from functions_expe_comparison import get_SMD_from_integrated_profile, get_SMD_flux_weighted, get_mean_diameters_from_global_spray
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
plt.rcParams['lines.markersize'] = 15*FFIG
plt.rcParams['legend.loc']       = 'best'
plt.rcParams['text.usetex'] = True

folder = 'C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/'
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch5_resolved_JICF/results_trajectories/'



label_x   = r'$x~[\mathrm{mm}]$'
label_y   = r'$y~[\mathrm{mm}]$'
label_z   = r'$z~[\mathrm{mm}]$'
label_ql  = r'$\langle q_l\rangle$ [$\mathrm{cm}^3/\mathrm{cm}^2\mathrm{s}$]'
label_SMD = r'$\langle SMD \rangle$ [$\mu\mathrm{m}$]'
label_SMD_global = r'$ SMD $ [$\mu\mathrm{m}$]'
label_expe  = r'$\mathrm{Experiments}$'
width_error_lines = 4*FFIG
caps_error_lines  = 15*FFIG


label_SPS_x05 = r'$\mathrm{SPS}, x=05~\mathrm{mm}$'
label_SPS_x10 = r'$\mathrm{SPS}, x=10~\mathrm{mm}$'

fmt_SPS_x05 = '--k'
fmt_SPS_x10 = '--b'


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


params_simulation = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 23.33,
                     'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 100,
                     'SIGMA': 22e-3,
                     'D_inj': 0.45e-3}
params_simulation['Q_inj'] = np.pi/4*params_simulation['D_inj']**2*params_simulation['U_L']

# estimate errors
error_SMD  = 0.26
error_flux = 0.37
error_SMD  = 0.14
error_flux = 0.2

error_q_y_expe = flux_y_exp*error_flux
error_q_z_expe = flux_z_exp*error_flux
error_SMD_y_expe = SMD_y_exp*error_SMD
error_SMD_z_expe = SMD_z_exp*error_SMD

SMD_expe = 31


#%% SPS spray UG100_DX10


params_simulation_UG75 = {'RHO_L': 795, 'MU_L': 1.5e-3, 'U_L'  : 17.5,
                          'RHO_G': 7.21, 'MU_G': 1.82e-5, 'U_G'  : 75,
                          'SIGMA': 22e-3,
                          'D_inj': 0.45e-3}
params_simulation_UG75['Q_inj'] = np.pi/4*params_simulation_UG75['D_inj']**2*params_simulation_UG75['U_L']

# Load sprays UG100_DX10
_, _, sp_SPS, _, _ = load_all_SPS_global_sprays(params_simulation_UG75, params_simulation)

sprays_list = [sp_SPS]

grids = load_all_SPS_grids(sprays_list)

SPS_sp_x05   = sp_SPS[0]
SPS_sp_x10   = sp_SPS[1]
SPS_grid_x05 = grids[0][0]
SPS_grid_x10 = grids[0][1]

# get global SMDs
x_SPS   = [5,10]
SMD_SPS = [SPS_sp_x05.SMD, SPS_sp_x10.SMD]

# average along y
z_prof_SPS_x05, ql_SPS_x05_prof_along_z, SMD_SPS_x05_prof_along_z = average_along_y(SPS_grid_x05)
z_prof_SPS_x10, ql_SPS_x10_prof_along_z, SMD_SPS_x10_prof_along_z = average_along_y(SPS_grid_x10)
# average along z
y_prof_SPS_x05, ql_SPS_x05_prof_along_y, SMD_SPS_x05_prof_along_y = average_along_z(SPS_grid_x05)
y_prof_SPS_x10, ql_SPS_x10_prof_along_y, SMD_SPS_x10_prof_along_y = average_along_z(SPS_grid_x10)

SMD_lim_SPS_along_y = (0,110)
SMD_lim_SPS_along_z = (0,135)

#%% LGS sprays                   

parent_dir = folder+"LGS_sprays_final"
#dirs       = ["xInj10mm/ALM_no_second_no" , "xInj10mm/ALM_no_second_yes", 
#              "xInj10mm/ALM_yes_second_no", "xInj10mm/ALM_yes_second_yes"]

dirs       = ["dx10_x02mm_wRMS"]
sols_dirs_name  = None
filename        = "vol_dist_coarse"  # Only applies if 'loadDropletsDistr = False'
sampling_planes = ['x = 06 mm', 'x = 07 mm', 'x = 08 mm', 'x = 09 mm',
                   'x = 10 mm','x = 11 mm', 'x = 12 mm', 'x = 13 mm', 'x = 14 mm',
                   'x = 15 mm', 'x = 16 mm', 'x = 17 mm', 'x = 18 mm', 'x = 19 mm', 
                   'x = 20 mm', 'x = 25 mm', 'x = 30 mm', 'x = 35 mm','x = 40 mm', 
                   'x = 45 mm', 'x = 55 mm', 'x = 60 mm', 'x = 80 mm']


# Get relevant LGS to plot maps
#i_LGS = [0,5,10,11,13,-1]
i_LGS = [4,14,-1]
format_LGS = ['b','r','y']
       
n_planes_LGS = len(sampling_planes)
            
dirs = [parent_dir+'/'+d for d in dirs]
sprays_list_LGS = get_sprays_list(True, sampling_planes, dirs, filename,
                                  params_simulation,
                                  sols_dirs_name = '.',
                                  D_outlier = 300000)

dirs_grid = [ [d] for d in dirs]
grids_list_LGS = get_discrete_spray(True, sprays_list_LGS, [8]*2, 
                                None, params_simulation ,
                                DIR = folder)

common_bounds = get_grids_common_boundaries(grids_list_LGS)


# get global and spatially integrated profiles
x_LGS = [5]; SMD_LGS = [SPS_sp_x05.SMD]; names_LGS = []
D0p1_LGS_all = []; D0p5_LGS_all = []; D0p9_LGS_all = []
z_prof_LGS_all = []; ql_prof_along_z_LGS_all = []; SMD_prof_along_z_LGS_all = []
y_prof_LGS_all = []; ql_prof_along_y_LGS_all = []; SMD_prof_along_y_LGS_all = []
for i in range(n_planes_LGS):
    spray_i = sprays_list_LGS[0][i]
    grid_i  = grids_list_LGS[0][i]
    
    # get global data
    x_i   = float(spray_i.name.split()[2])
    if x_i < 15:
        SMD_i = spray_i.SMD 
    else:
        SMD_i = get_SMD_flux_weighted(grid_i)
    SMD_i = spray_i.SMD 
    D0p1, D0p5, D0p9 = get_mean_diameters_from_global_spray(spray_i)   
    

    # append global data
    names_LGS.append(r'$\mathrm{LGS}, x = '+str(int(x_i))+r' \mathrm{mm}$')
    D0p1_LGS_all.append(D0p1)
    D0p5_LGS_all.append(D0p5)
    D0p9_LGS_all.append(D0p9)
    x_LGS.append(x_i)
    SMD_LGS.append(SMD_i)
    
    
    # get integrated profiles    
    z_prof_i, ql_prof_along_z, SMD_prof_along_z = average_along_y(grid_i)
    y_prof_i, ql_prof_along_y, SMD_prof_along_y = average_along_z(grid_i)
    
    # append integrated profiles
    z_prof_LGS_all.append(z_prof_i)
    ql_prof_along_z_LGS_all.append(ql_prof_along_z)
    SMD_prof_along_z_LGS_all.append(SMD_prof_along_z)
    y_prof_LGS_all.append(y_prof_i)
    ql_prof_along_y_LGS_all.append(ql_prof_along_y)
    SMD_prof_along_y_LGS_all.append(SMD_prof_along_y)










#%% Compare SPS (x = 5, 10 mm) vs expe (x = 80 mm)

# along z
plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title("SPS vs expe: flux")
plt.plot(flux_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(flux_z_exp, z_int_exp, xerr=error_q_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(ql_SPS_x05_prof_along_z, z_prof_SPS_x05, 'k', label=label_SPS_x05)
plt.plot(ql_SPS_x10_prof_along_z, z_prof_SPS_x10, 'b', label=label_SPS_x10)
plt.legend(loc='best')
plt.xlabel(label_ql)
plt.ylabel(label_z)
plt.show()
plt.close()

plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title("SPS vs expe: SMD")
plt.plot(SMD_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(SMD_z_exp, z_int_exp, xerr=error_SMD_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(SMD_SPS_x05_prof_along_z, z_prof_SPS_x05, 'k', label=label_SPS_x05)
plt.plot(SMD_SPS_x10_prof_along_z, z_prof_SPS_x10, 'b', label=label_SPS_x10)
plt.legend(loc='best')
plt.xlabel(label_SMD)
plt.ylabel(label_z)
plt.show()
plt.close()


#along y
plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title("SPS vs expe: flux")
plt.plot(y_int_exp, flux_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, flux_y_exp, yerr=error_q_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(y_prof_SPS_x05, ql_SPS_x05_prof_along_y, 'k', label=label_SPS_x05)
plt.plot(y_prof_SPS_x10, ql_SPS_x10_prof_along_y, 'b', label=label_SPS_x10)
plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_ql)
plt.show()
plt.close()

plt.figure(figsize=(FFIG*18,FFIG*13))
plt.title("SPS vs expe: SMD")
plt.plot(y_int_exp, SMD_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, SMD_y_exp, yerr=error_SMD_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(y_prof_SPS_x05, SMD_SPS_x05_prof_along_y, 'k', label=label_SPS_x05)
plt.plot(y_prof_SPS_x10, SMD_SPS_x10_prof_along_y, 'b', label=label_SPS_x10)
plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_SMD)
plt.show()
plt.close()





#%% SMD evolution along x


plt.figure(figsize=(FFIG*20,FFIG*10))
plt.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
plt.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(x_SPS, SMD_SPS, '-ok', label=r'$\mathrm{SPS}$')
plt.plot(x_LGS, SMD_LGS, '-ob', label=r'$\mathrm{LGS}$')
plt.xlim(4,81)
plt.xlabel(label_x)
plt.ylabel(label_SMD_global)
plt.xticks([5,10,20,30,40,50,60,70,80])
plt.ylim(0,90)
plt.grid()
plt.legend(loc='best', numpoints = 2, framealpha=1)
plt.tight_layout()
plt.show()
plt.close()

# Zoom in 75 < x < 85
plt.figure(figsize=(FFIG*17,FFIG*10))
plt.scatter(80,SMD_expe,color='black',marker='s',label=r'$\mathrm{Expe}$')
plt.errorbar(80, SMD_expe, yerr=SMD_expe*error_SMD, color='black', fmt='s',
             markersize=30*FFIG,linewidth=width_error_lines,capsize=30*FFIG)
plt.plot(x_LGS[-1], SMD_LGS[-1], '-ob', label=r'$\mathrm{LGS}$',markersize=30*FFIG)
plt.xlim(75,85)
plt.xlabel(label_x)
plt.ylabel(label_SMD_global)
plt.xticks([80])
plt.ylim(15,40)
plt.grid()
#plt.legend(loc='best', numpoints = 2, framealpha=1)
plt.tight_layout()
plt.show()
plt.close()



#%% LGS vs expe all

figsize_LGS_all_curves = (FFIG*25,FFIG*20)

# along z
plt.figure(figsize=figsize_LGS_all_curves)
plt.title("SPS vs LGS: flux")
plt.plot(flux_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(flux_z_exp, z_int_exp, xerr=error_q_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(n_planes_LGS):
    if i%2 == 0:
        fmt = '-'
    else:
        fmt = '--'
    plt.plot(ql_prof_along_z_LGS_all[i], z_prof_LGS_all[i], fmt,label=names_LGS[i])
plt.legend(loc='best', bbox_to_anchor=(1.01,1))
plt.xlabel(label_ql)
plt.ylabel(label_z)
plt.show()
plt.close()

plt.figure(figsize=figsize_LGS_all_curves)
plt.title("SPS vs LGS: SMD")
plt.plot(SMD_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(SMD_z_exp, z_int_exp, xerr=error_SMD_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(n_planes_LGS):
    if i%2 == 0:
        fmt = '-'
    else:
        fmt = '--'
    plt.plot(SMD_prof_along_z_LGS_all[i],z_prof_LGS_all[i], fmt,label=names_LGS[i])
plt.legend(loc='best', bbox_to_anchor=(1.01,1))
plt.xlabel(label_SMD)
plt.ylabel(label_z)
plt.show()
plt.close()



#along y
plt.figure(figsize=figsize_LGS_all_curves)
plt.title("SPS vs expe: flux")
plt.plot(y_int_exp, flux_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, flux_y_exp, yerr=error_q_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(n_planes_LGS):
    if i%2 == 0:
        fmt = '-'
    else:
        fmt = '--'
    plt.plot(y_prof_LGS_all[i], ql_prof_along_y_LGS_all[i], fmt, label=names_LGS[i])
plt.legend(loc='best', bbox_to_anchor=(1.01,1))
plt.xlabel(label_y)
plt.ylabel(label_ql)
plt.show()
plt.close()

plt.figure(figsize=figsize_LGS_all_curves)
plt.title("SPS vs expe: SMD")
plt.plot(y_int_exp, SMD_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, SMD_y_exp, yerr=error_SMD_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(n_planes_LGS):
    if i%2 == 0:
        fmt = '-'
    else:
        fmt = '--'
    plt.plot(y_prof_LGS_all[i], SMD_prof_along_y_LGS_all[i], fmt, label=names_LGS[i])
plt.legend(loc='best', bbox_to_anchor=(1.01,1))
plt.xlabel(label_y)
plt.ylabel(label_SMD)
plt.show()
plt.close()


#%% LGS vs SPS vs expe

names_LGS_to_plot  = []
z_prof_LGS_to_plot = []; ql_prof_along_z_LGS_to_plot = []; SMD_prof_along_z_LGS_to_plot = []
y_prof_LGS_to_plot = []; ql_prof_along_y_LGS_to_plot = []; SMD_prof_along_y_LGS_to_plot = []
for i in i_LGS:
    names_LGS_to_plot.append(names_LGS[i])
    
    z_prof_LGS_to_plot.append(z_prof_LGS_all[i])
    ql_prof_along_z_LGS_to_plot.append(ql_prof_along_z_LGS_all[i])
    SMD_prof_along_z_LGS_to_plot.append(SMD_prof_along_z_LGS_all[i])
    
    y_prof_LGS_to_plot.append(y_prof_LGS_all[i])
    ql_prof_along_y_LGS_to_plot.append(ql_prof_along_z_LGS_all[i])
    SMD_prof_along_y_LGS_to_plot.append(SMD_prof_along_z_LGS_all[i])
    


# along z
plt.figure(figsize=(FFIG*18,FFIG*13))
#plt.title("SPS vs expe: flux")
plt.plot(flux_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(flux_z_exp, z_int_exp, xerr=error_q_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(ql_SPS_x05_prof_along_z, z_prof_SPS_x05, fmt_SPS_x05, label=label_SPS_x05)
plt.plot(ql_SPS_x10_prof_along_z, z_prof_SPS_x10, fmt_SPS_x10, label=label_SPS_x10)
for i in range(len(names_LGS_to_plot)):
    plt.plot(ql_prof_along_z_LGS_to_plot[i], z_prof_LGS_to_plot[i], 
             format_LGS[i], label=names_LGS_to_plot[i])
plt.legend(loc='best')
plt.xlabel(label_ql)
plt.ylabel(label_z)
plt.grid()
plt.show()
plt.close()

plt.figure(figsize=(FFIG*18,FFIG*13))
#plt.title("SPS vs expe: SMD")
plt.plot(SMD_z_exp, z_int_exp, 'ks', label=label_expe)
plt.errorbar(SMD_z_exp, z_int_exp, xerr=error_SMD_z_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
for i in range(len(names_LGS_to_plot)):
    plt.plot(SMD_prof_along_z_LGS_to_plot[i], z_prof_LGS_to_plot[i], 
             format_LGS[i], label=names_LGS_to_plot[i])
plt.plot(SMD_SPS_x05_prof_along_z, z_prof_SPS_x05, fmt_SPS_x05, label=label_SPS_x05)
plt.plot(SMD_SPS_x10_prof_along_z, z_prof_SPS_x10, fmt_SPS_x10, label=label_SPS_x10)
#plt.legend(loc='best')
plt.xlabel(label_SMD)
plt.xlim(SMD_lim_SPS_along_z)
plt.ylabel(label_z)
plt.grid()
plt.show()
plt.close()



#along y
plt.figure(figsize=(FFIG*18,FFIG*13))
#plt.title("SPS vs expe: flux")
plt.plot(y_int_exp, flux_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, flux_y_exp, yerr=error_q_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(y_prof_SPS_x05, ql_SPS_x05_prof_along_y, fmt_SPS_x05, label=label_SPS_x05)
plt.plot(y_prof_SPS_x10, ql_SPS_x10_prof_along_y, fmt_SPS_x10, label=label_SPS_x10)
for i in range(len(names_LGS_to_plot)):
    plt.plot(y_prof_LGS_to_plot[i], ql_prof_along_y_LGS_to_plot[i], 
             format_LGS[i], label=names_LGS_to_plot[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_ql)
plt.grid()
plt.show()
plt.close()

plt.figure(figsize=(FFIG*18,FFIG*13))
#plt.title("SPS vs expe: SMD")
plt.plot(y_int_exp, SMD_y_exp, 'ks', label=label_expe)
plt.errorbar(y_int_exp, SMD_y_exp, yerr=error_SMD_y_expe, color='black', fmt='o',
             linewidth=width_error_lines,capsize=caps_error_lines)
plt.plot(y_prof_SPS_x05, SMD_SPS_x05_prof_along_y, fmt_SPS_x05, label=label_SPS_x05)
plt.plot(y_prof_SPS_x10, SMD_SPS_x10_prof_along_y, fmt_SPS_x10, label=label_SPS_x10)
for i in range(len(names_LGS_to_plot)):
    plt.plot(y_prof_LGS_to_plot[i], SMD_prof_along_y_LGS_to_plot[i], 
             format_LGS[i], label=names_LGS_to_plot[i])
#plt.legend(loc='best')
plt.xlabel(label_y)
plt.ylabel(label_SMD)
plt.ylim(SMD_lim_SPS_along_y)
plt.grid()
plt.show()
plt.close()


