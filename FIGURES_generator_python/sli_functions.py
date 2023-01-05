"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""

import matplotlib.pyplot as plt
from sprPost_calculations import get_sprays_list, get_discrete_spray
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')



def load_all_BIMER_global_sprays(params_simulation, sampling_planes = None):
    
    # General keywords
    parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/BIMER_SPS_sprays"
    filename   = "vol_dist_plane"
    
    # BIMER
    if sampling_planes is None:
        sampling_planes = ['xD_03p33','xD_05p00','xD_06p67']
    
    
    dirs_DX15 = ["dx15p0"]
    dirs_DX10 = ["dx10p0"]
    dirs_DX07 = ["dx07p5"]
    
    dirs_DX15 = [parent_dir +'/'+ d for d in dirs_DX15]
    dirs_DX10 = [parent_dir +'/'+ d for d in dirs_DX10]
    dirs_DX07 = [parent_dir +'/'+ d for d in dirs_DX07]
    
    # DX15
    sprays_list_DX15 = get_sprays_list(True, sampling_planes, 
                                       dirs_DX15, 
                                       filename,
                                       params_simulation,
                                       CASE = 'BIMER',
                                       sols_dirs_name = ".")
    sprays_list_DX15 = sprays_list_DX15[0]
    
    # DX10
    sprays_list_DX10 = get_sprays_list(True, sampling_planes, 
                                       dirs_DX10, 
                                       filename,
                                       params_simulation,
                                       CASE = 'BIMER',
                                       sols_dirs_name = ".")
    sprays_list_DX10 = sprays_list_DX10[0]
    
    # DX07
    sprays_list_DX07 = get_sprays_list(True, sampling_planes, 
                                       dirs_DX07, 
                                       filename,
                                       params_simulation,
                                       CASE = 'BIMER',
                                       sols_dirs_name = ".")
    sprays_list_DX07 = sprays_list_DX07[0]
    
    return sprays_list_DX07, sprays_list_DX10, sprays_list_DX15

    

def load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100,
                               parent_dir = None,
                               save_dir = 'store_variables'):    
    
    # General keywords
    filename   = "vol_dist_coarse"
    if parent_dir is None:
        parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/SPS_sprays"
    
    # JICF SPS solutions 
    filename        = "vol_dist_coarse"
    sampling_planes_DX10 = ['x = 05 mm', 'x = 10 mm']
    sampling_planes_DX20 = ['x = 05 mm', 'x = 10 mm', 'x = 15 mm']
    
    
    dirs_UG75_DX10 = ["q6uG75/sps_dx=10m"]
    dirs_UG75_DX20 = ["q6uG75/sps_dx=20m"]
    dirs_UG100_DX10 = ["q6uG100/sps_dx=10m"]
    dirs_UG100_DX20 = ["q6uG100/sps_dx=20m"]
    dirs_UG100_DX20_NT = ["q6uG100/sps_dx=20m_no_turb" ]
    
    dirs_UG75_DX10 = [parent_dir +'/'+ d for d in dirs_UG75_DX10]
    dirs_UG75_DX20 = [parent_dir +'/'+ d for d in dirs_UG75_DX20]
    dirs_UG100_DX10 = [parent_dir +'/'+ d for d in dirs_UG100_DX10]
    dirs_UG100_DX20 = [parent_dir +'/'+ d for d in dirs_UG100_DX20]
    dirs_UG100_DX20_NT = [parent_dir +'/'+ d for d in dirs_UG100_DX20_NT]
    
    # UG75_DX10
    sprays_list_UG75_DX10 = get_sprays_list(True, sampling_planes_DX10, 
                                             dirs_UG75_DX10, 
                                             filename,
                                             params_simulation_UG75,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".",
                                             save_dir = save_dir)
    sprays_list_UG75_DX10 = sprays_list_UG75_DX10[0]
    
    # UG75_DX20
    sprays_list_UG75_DX20 = get_sprays_list(True, sampling_planes_DX20, 
                                             dirs_UG75_DX20, 
                                             filename,
                                             params_simulation_UG75,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".",
                                             save_dir = save_dir)
    sprays_list_UG75_DX20 = sprays_list_UG75_DX20[0]
    
    
    # UG100_DX10
    sprays_list_UG100_DX10 = get_sprays_list(True, sampling_planes_DX10, 
                                             dirs_UG100_DX10, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".",
                                             save_dir = save_dir)
    sprays_list_UG100_DX10 = sprays_list_UG100_DX10[0]
    
    # UG100_DX20
    sprays_list_UG100_DX20 = get_sprays_list(True, sampling_planes_DX20, 
                                             dirs_UG100_DX20, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".",
                                             save_dir = save_dir)
    sprays_list_UG100_DX20 = sprays_list_UG100_DX20[0]
    
    # UG100_DX20_NT
    sprays_list_UG100_DX20_NT = get_sprays_list(True, sampling_planes_DX10, 
                                             dirs_UG100_DX20_NT, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".",
                                             save_dir = save_dir)
    sprays_list_UG100_DX20_NT = sprays_list_UG100_DX20_NT[0]
    
    return sprays_list_UG75_DX10, sprays_list_UG75_DX20, sprays_list_UG100_DX10, sprays_list_UG100_DX20, sprays_list_UG100_DX20_NT 



    
def load_all_BIMER_grids(sprays_list, parent_dir = None, save_dir = 'store_variables'):   
    
    if parent_dir is None:
        parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing"
    

    grids_list = get_discrete_spray(True, sprays_list, None, None, None,
                                    DIR = parent_dir, save_dir = save_dir,
                                    CASE = 'BIMER')
    
    return grids_list


def load_all_SPS_grids(sprays_list, parent_dir = None, save_dir = 'store_variables'):   
    
    if parent_dir is None:
        parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/"
    

    grids_list = get_discrete_spray(True, sprays_list, None, None, None,
                                    DIR = parent_dir, save_dir = save_dir)
    
    return grids_list



         
def plot_grid(grid, ADD_TO_FIGURE = True, PLOT_CENTER = False, PLOT_QUADTREES = False):
    if not ADD_TO_FIGURE:
        plt.figure(figsize=(20,15))
        if PLOT_CENTER:
            plt.plot(grid.yy_center, grid.zz_center, marker='.', 
                     markersize=100/min(grid.grid_size), color='k', linestyle='None')

    for i in range(0,grid.grid_size[0] + 1):
        plt.plot([grid.y[i]]*2,[grid.bounds[1][0], grid.bounds[1][1]], 
                 linewidth=3, color='k', alpha=0.5)
    # Plot now horizontal lines, so we need to iterate between z lines
    for j in range(0,grid.grid_size[1] + 1):
        plt.plot([grid.bounds[0][0], grid.bounds[0][1]], [grid.z[j]]*2, 
                 linewidth=3, color='k', alpha=0.5)
        
    # If quadtrees, refine grid first level
    for m in range(grid.grid_size[1]):
        for n in range(grid.grid_size[0]):
            if grid.QUADTREES[m][n] and PLOT_QUADTREES:
                child_x02_grid = grid.SPRAYS_array[m][n]
                for i in range(0,child_x02_grid.grid_size[0] + 1):
                    plt.plot([child_x02_grid.y[i]]*2,[child_x02_grid.bounds[1][0], child_x02_grid.bounds[1][1]], 
                             linewidth=3, color='k')
                for j in range(0,child_x02_grid.grid_size[1] + 1):
                    plt.plot([child_x02_grid.bounds[0][0], child_x02_grid.bounds[0][1]], [child_x02_grid.z[j]]*2,
                             linewidth=3, color='k')
                    
                #  If quadtrees, refine grid second level 
                for m_ch in range(child_x02_grid.grid_size[0]):
                    for n_ch in range(child_x02_grid.grid_size[1]):
                        if child_x02_grid.QUADTREES[m_ch][n_ch]:
                            child_x04_grid = child_x02_grid.SPRAYS_array[m_ch][n_ch]
                            for i in range(0,child_x04_grid.grid_size[0] + 1):
                                plt.plot([child_x04_grid.y[i]]*2,[child_x04_grid.bounds[1][0], child_x04_grid.bounds[1][1]], 
                                         linewidth=3, color='k')
                            for j in range(0,child_x04_grid.grid_size[1] + 1):
                                plt.plot([child_x04_grid.bounds[0][0], child_x04_grid.bounds[0][1]], [child_x04_grid.z[j]]*2,
                                         linewidth=3, color='k')
                    
                
    
    if not ADD_TO_FIGURE:
        plt.title("The grid", fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel('y [mm]',fontsize=30)
        plt.ylabel('z [mm]',fontsize=30)
        plt.show()
        
def plot_grid_highlight_probe(grid, m, n, color='blue'):
    
    
    # Plot vertical lines
    for i in range(m, m+2):
        plt.plot([grid.y[i]]*2,[grid.z[n],grid.z[n+1]], 
                 linewidth=6, color=color)
    # Plot  horizontal lines
    for j in range(n, n+ 2):
        plt.plot([grid.y[m],grid.y[m+1]], [grid.z[j]]*2, 
                 linewidth=6, color=color)
    
   
            