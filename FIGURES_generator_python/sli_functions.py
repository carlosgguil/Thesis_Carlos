"""
Created on Mon Jul  1 10:04:13 2019

@author: Carlos G. GUILLAMON
"""


from sprPost_calculations import get_sprays_list
import sys
sys.path.append('C:/Users/Carlos Garcia/Documents/GitHub/spr_post')

def load_all_SPS_global_sprays(params_simulation_UG75, params_simulation_UG100):    
    
    # General keywords
    parent_dir = "C:/Users/Carlos Garcia/Desktop/Ongoing/Droplet postprocessing/SPS_sprays"
    filename   = "vol_dist_coarse"
    
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
                                             sols_dirs_name = ".")
    sprays_list_UG75_DX10 = sprays_list_UG75_DX10[0]
    
    # UG75_DX20
    sprays_list_UG75_DX20 = get_sprays_list(True, sampling_planes_DX20, 
                                             dirs_UG75_DX20, 
                                             filename,
                                             params_simulation_UG75,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".")
    sprays_list_UG75_DX20 = sprays_list_UG75_DX20[0]
    
    
    # UG100_DX10
    sprays_list_UG100_DX10 = get_sprays_list(True, sampling_planes_DX10, 
                                             dirs_UG100_DX10, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".")
    sprays_list_UG100_DX10 = sprays_list_UG100_DX10[0]
    
    # UG100_DX20
    sprays_list_UG100_DX20 = get_sprays_list(True, sampling_planes_DX20, 
                                             dirs_UG100_DX20, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".")
    sprays_list_UG100_DX20 = sprays_list_UG100_DX20[0]
    
    # UG100_DX20_NT
    sprays_list_UG100_DX20_NT = get_sprays_list(True, sampling_planes_DX10, 
                                             dirs_UG100_DX20_NT, 
                                             filename,
                                             params_simulation_UG100,
                                             CASE = 'JICF',
                                             sols_dirs_name = ".")
    sprays_list_UG100_DX20_NT = sprays_list_UG100_DX20_NT[0]
    
    return sprays_list_UG75_DX10, sprays_list_UG75_DX20, sprays_list_UG100_DX10, sprays_list_UG100_DX20, sprays_list_UG100_DX20_NT 

    




            
            