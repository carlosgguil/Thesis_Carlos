# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:52:19 2020

@author: Carlos García
"""
from copy import deepcopy
from operator import itemgetter
from scipy.interpolate import interp1d
import os, os.path
import pandas as pd
import numpy as np
import pickle
import re


# ---------------------------------------
# FUNCTION calculate_L2_error
#    
def calculate_L2_error(x2_exp, x2_num):
    """
    Calculates the L2 error for a trajectory (vertical or lateral)
    """
    N = len(x2_exp)
    
    assert (N == len(x2_num)), 'x2_exp and x2_num vectors do not have same length !'

    L2 = 0
    for i in range(N):
        L2 += (x2_num[i] - x2_exp[i])**2
    L2 = np.sqrt(L2/N)
    
    return L2

# 
# END FUNCTION calculate_L2_error 
# --------------------------------------- 


# ---------------------------------------
# FUNCTION get_instantaneous_trajectory
#    
def get_inst_traj(x2_bins, x1_val_time, x2_val_time, filter_x2_max = False):
    '''
        Function 'get_instantaneous_trajectory'.
        Processes raw data x1_val_time, x2_val_time and returns the instantaneous
        trajectories in the x2 range given by x2_bins
    '''
    x2_min = x2_bins[0]
    nBins = len(x2_bins) - 1
    Nt    = len(x1_val_time) # Nt is the number of time instants    
    x1_inst_trajectories    = [ np.ones(nBins)*np.nan for i in range(Nt)] 
    x2_inst_trajectories    = [ np.ones(nBins)*np.nan for i in range(Nt)] 
            
    k = 0
    for k in range(Nt):
                           
        # Select time step points
        x1_val_tk = x1_val_time[k]
        x2_val_tk = x2_val_time[k]
        # Order timestep by z values
        x2_val_tk, x1_val_tk = range_vector_pairs(x2_val_tk, x1_val_tk)
        
        i = 0
        for n in range(nBins):
            x2_low = x2_bins[n]
            x2_upp = x2_bins[n+1]
            
            x1_class = []
            x2_class = []
    
            x2val_in_bin = True
            while x2val_in_bin:
                if i >= len(x2_val_tk):
                    break
                if x2_val_tk[i] < x2_min: # Case x2_value is lower than 0
                    i += 1
                elif (x2_val_tk[i] >= x2_low) and (x2_val_tk[i] < x2_upp):
                    x1_class.append(x1_val_tk[i])
                    x2_class.append(x2_val_tk[i])
                    i += 1
                else:
                    x2val_in_bin = False
                    
            if len(x2_class) != 0:
                x1_class, x2_class = range_vector_pairs(x1_class, x2_class)
                x1_inst_trajectories[k][n] = x1_class[0]# + D/2
                x2_inst_trajectories[k][n] = x2_class[0]
        
        # Remove the nans
        nan_array_x1 = np.isnan(x1_inst_trajectories[k])
        nan_array_x2 = np.isnan(x2_inst_trajectories[k])
        non_nan_array_x1 = ~ nan_array_x1
        non_nan_array_x2 = ~ nan_array_x2
        x1_inst_trajectories[k] = x1_inst_trajectories[k][non_nan_array_x1]
        x2_inst_trajectories[k] = x2_inst_trajectories[k][non_nan_array_x2]
        
        # Range by x1 coordinate
        a, b = range_vector_pairs(x1_inst_trajectories[k], x2_inst_trajectories[k])
          
        x1 = a
        x2 = b
        
        if filter_x2_max:
            # Sweep x1 axis and neglect points whose previous liquid structure is lower
            x1 = [a[0]]
            x2 = [b[0]]
            x2_max = 0
            for m in range(1,len(a)):
                if (b[m] > b[m-1]) and (b[m] > x2_max):
                    x2_max = b[m] 
                    x1.append(a[m])
                    x2.append(b[m])
        
        x1_inst_trajectories[k], x2_inst_trajectories[k] = np.array(x1), np.array(x2)
        # Shift x1 vector to start at x = 0 (if vector is not empty)
        if x1_inst_trajectories:
            x1_inst_trajectories[k] -= x1_inst_trajectories[k][0]
        else:
            print(f'WARNINGS: trajectory {k} is empty')
        
    return [x1_inst_trajectories, x2_inst_trajectories]
# 
# END FUNCTION get_inst_traj 
# --------------------------------------- 

# ---------------------------------------
# FUNCTION get_presence_rate
#    
def get_presence_rate(x_int_range, df_x_inst_trajectories, Nt):
    '''
    Function to get liquid presence rate along x axis
    '''
    
    x_min_int = x_int_range[0]
    x_max_int = x_int_range[-1]
    dx = np.diff(x_int_range)[0]
    
    df_presence_rate = pd.DataFrame()
    for i in range(Nt):
        presence_rate_i = np.zeros(len(x_int_range))
        presence_rate_i[0] = 1
        for j in df_x_inst_trajectories.iloc[1:,i]:
            if (not np.isnan(j)) and (j < x_max_int):
                index = int(np.floor((j-x_min_int)/dx))
                presence_rate_i[index+1] = 1
        df_presence_rate['t_'+str(i+1)] = presence_rate_i
    df_presence_rate['t_mean'] = df_presence_rate.mean(axis=1)
    
    return df_presence_rate
    
# END FUNCTION get_presence_rate 
# --------------------------------------- 

# ---------------------------------------
# FUNCTION interpolate_instantaneous_trajectories
#    
def interp_inst_traj(x1_int_range, Nt, D, 
                     df_x1_inst_trajectories, 
                     df_x2_inst_trajectories):
    '''
    Function for interpolating instantenous trajectories
    Takes the instantaneous trajectories dataframe and interpolates them in the
    range given by 'x1_int_range'.
    Returns four dataframes:
        'df_x1_int' with x of all instantaneous trajectories
        'df_x2_int' with y or z coordinates of all instantaneous trajectories
        'df_mean_temporal' with x1 and x2 coordinates of the time-dependant mean trajectory 
        'df_mean' with x1 and x2 coordinates of the final mean trajectory
    '''
    
    x1_max_int = x1_int_range[-1]
    x2_int_columns = [[]]*Nt
    for i in range(Nt):
        # 1 Selection of x and z values from simulation (i.e. no interpolated)
        x1_noInt_i = df_x1_inst_trajectories.iloc[:,i].dropna().tolist()
        x2_noInt_i = df_x2_inst_trajectories.iloc[:,i].dropna().tolist()
        # 2 Interpolate a function f 
        f = interp1d(x1_noInt_i, x2_noInt_i)
        # 3 Apply function f to the range x_int_range to get the interpolated z values
        # Condition: instantaneous trajectory must be within x range for interpolation
        if x1_noInt_i[-1] >= x1_max_int: 
            # If True, then interpolate in x range
            x2_int_i = f(x1_int_range)
        else:
            # If false, then reduce x range for interpolation 
            x2_int_i = np.ones(len(x1_int_range))*np.nan
            j = 0
            while x1_int_range[j] <= x1_noInt_i[-1]:
                j += 1
            x2_int_i[:j] = f(x1_int_range[:j])
        # 4 Store these values in 
        x2_int_columns[i] = x2_int_i
        
    # Create x dataframe with interpolated values
    df_x1_int = pd.DataFrame(x1_int_range)
    df_x1_int = df_x1_int.T
    # Create new z dataframe with interpolated values
    df_x2_int = pd.DataFrame(x2_int_columns)
    df_x2_int = df_x2_int.T
    
    # Calculate the temporal evolution of the trajectory
    df_mean_temporal = pd.DataFrame()
    df_mean_temporal['x1D']   = x1_int_range/D
    df_mean_temporal['x2D_1'] = df_x2_int[0]/D
    for i in range(1,Nt):
        index = 'x2D_' + str(i+1)
        df_mean_temporal[index] = df_x2_int.iloc[:,0:i+1].mean(axis=1)/D
    
    
    
    ##########################################################
    ####       Part III: final trajectory obtention  #########
    ##########################################################
        
    df_mean = pd.DataFrame()
    df_mean['x1'] = x1_int_range 
    df_mean['x2'] = df_x2_int.mean(axis=1)
    df_mean['x1/D'] = df_mean['x1']/D
    df_mean['x2/D'] = df_mean['x2']/D
           
    return [df_x1_int, df_x2_int, df_mean_temporal, df_mean]
# 
# END FUNCTION interp_inst_traj 
# --------------------------------------- 

  





# ---------------------------------------
# FUNCTION pickleLoad
#    
def pickleLoad(variableFile):
    with open(variableFile, 'rb') as f:
        variable = pickle.load(f)
    return variable
# 
# END FUNCTION pickleLoad 
# --------------------------------------- 




# ---------------------------------------
# FUNCTION pickleSave
#    
def pickleSave(variable, variableFile):
    with open(variableFile, 'wb') as f:
        pickle.dump(variable, f)
# 
# END FUNCTION pickleSave
# --------------------------------------- 




# ---------------------------------------
# FUNCTION process_dataFrame_notInterpolated
#    
def process_MC_dataFrame_notInterpolated(x_val_MC_time, z_val_MC_time, file_to_store):
    # Used in approach 1 (so not useful)
    
    columns_x = []; columns_z = []
    for i in range(len(x_val_MC_time)):
        # a and b are vectors in time i 
        a = x_val_MC_time[i]   ; b = z_val_MC_time[i]
        a_order, b_order = range_vector_pairs(a, b)

        columns_x.append(a_order)
        columns_z.append(b_order)
    
    
        
    # Create pandas dataframe 
    df_x = pd.DataFrame(columns_x)
    df_x = df_x.T
    df_x = df_x.T.dropna(how='all').reset_index().T # Eliminate empty columns
    df_x.to_csv(file_to_store+'_x_MC_noInt.csv', index = False)
    df_z = pd.DataFrame(columns_z)
    df_z = df_z.T
    df_z = df_z.T.dropna(how='all').reset_index().T
    df_z.to_csv(file_to_store+'_z_MC_noInt.csv', index = False)    
    
    # Finally, let's drop all the columns of the dataframes with all elements missing
    
    return df_x, df_z
    
# 
# END FUNCTION process_dataFrame_notInterpolated
# ---------------------------------------         
        
        
# ---------------------------------------
# FUNCTION process_raw_data
#    
def process_raw_data(inputs, traj_type = []):
    index_init = inputs[0] 
    index_end  = inputs[1]
    case       = inputs[2]
    D          = inputs[3]
    
    n_sols = index_end - index_init + 1
    
    # Select coordinate y or z depending on trajectory type
    if traj_type == 'lateral':
        coord_x2 = 'Points:1'
    else:
        coord_x2 = 'Points:2'
    
    # Create a list with the names of all the folders
    folders_list = []
    indexes_list = []
    if n_sols != 1:
        for i in range(index_init, index_end+1):
            if len(str(i)) == 1:
               index = '0'+str(i)
            elif len(str(i)) == 2:
               index = str(i)
            folders_list.append(case+'/trajectory_'+index)
            indexes_list.append(index)
    else:
        if len(str(index_init)) == 1:
            index = '0'+str(index_init)
        elif len(str(index_init)) == 2:
            index = str(index_init)
        folders_list.append(case+'/trajectory_'+index)
        indexes_list.append(index)
    
    # Initialise lists:
    #    x1 is coordinate x
    #    x2 is coordinate y for lateral trajectory and z for vertical one
    x1_val = [] ; x1_val_time = []; x1_val_MC = [] ; x1_val_MC_time = [] 
    x2_val = [] ; x2_val_time = []; x2_val_MC = [] ; x2_val_MC_time = [] 
    
    # Loop on the folders
    for n in range(len(folders_list)):
        
        # Check if current folder solutions have already been processed
        file_data_trajectory_folder = folders_list[n]+'/data_trajectory_processed'
        
        if os.path.isfile(file_data_trajectory_folder):
            
            print(' Loaded data for '+case.split('/')[-1]+', '+folders_list[n].split('/')[-1])
            data_trajectory_folder = pickleLoad(file_data_trajectory_folder)
            
            x1_val_folder         = data_trajectory_folder[0]
            x1_val_time_folder    = data_trajectory_folder[1]
            x1_val_MC_folder      = data_trajectory_folder[2]
            x1_val_MC_time_folder = data_trajectory_folder[3]
            x2_val_folder         = data_trajectory_folder[4]
            x2_val_time_folder    = data_trajectory_folder[5]
            x2_val_MC_folder      = data_trajectory_folder[6]
            x2_val_MC_time_folder = data_trajectory_folder[7]
            
        else:
                
            # Obtain paths and number of files
            path, dirs, files = next(os.walk(folders_list[n]))
            n_files = len(files)
            
    
            # Obtain prefix by means of regular expressions
            csv_file_found = False
            counter = 0
            while (not csv_file_found):
                s = files[counter]
                s_format = s.split('.')[-1]
                if s_format == 'csv':
                    csv_file_found = True
                else:
                    counter += 1
            pattern = r'\d{1,3}.\d{1,3}.csv$'
            obj     = re.search(pattern, s)
            assert obj, 'Bad file naming: check it, or change regular expression pattern'
            prefix  = s[0:obj.start()]
            
            
            # count the time instants ---------------------------------------
            time_max     = -1
            lig_first    = prefix+'0'
            for file in files:
                file_split_dot = file.split('.')
                lig_current    = file_split_dot[0]
                if lig_current == lig_first:
                    time_current = file_split_dot[1]
                    if int(time_current) > int(time_max):
                        time_max = time_current
            # ---------------------------------------------------------------------
            
            # Calculate number of time instants and of ligaments within folder
            n_time_instants = int(time_max) + 1
            n_ligaments     = int(round(n_files/n_time_instants))
            
            
            
            
            # Loop on the files chronologically
            x1_val_folder         = [] 
            x1_val_MC_folder      = []  
            x1_val_time_folder    = [ [] for i in range(n_time_instants)]
            x1_val_MC_time_folder = [ [] for i in range(n_time_instants)]
                        
            x2_val_folder         = [] 
            x2_val_MC_folder      = []     
            x2_val_time_folder    = [ [] for i in range(n_time_instants)]
            x2_val_MC_time_folder = [ [] for i in range(n_time_instants)]
    
            
            for i in range(n_time_instants):
                if len(str(i)) == 1:
                    index_i = '0'+str(i)
                elif len(str(i)) == 2:
                    index_i = str(i)
                file_data_time_i = folders_list[n]+'/data_trajectory_time_'+index_i

                if os.path.isfile(file_data_time_i):

                    print(' Loaded time instant '+str(i)+' from '+case.split('/')[-1]+', '+folders_list[n].split('/')[-1]) 
                    data_trajectory_time_i   = pickleLoad(file_data_time_i)
                    x1_val_time_folder[i]    = data_trajectory_time_i[0] 
                    x1_val_MC_time_folder[i] = data_trajectory_time_i[1]
                    x2_val_time_folder[i]    = data_trajectory_time_i[2]
                    x2_val_MC_time_folder[i] = data_trajectory_time_i[3]
                    
                    x1_val_folder    = np.append(x1_val_folder, x1_val_time_folder[i])
                    x1_val_MC_folder = np.append(x1_val_MC_folder, x1_val_MC_time_folder[i])
                    x2_val_folder    = np.append(x2_val_folder, x2_val_time_folder[i])
                    x2_val_MC_folder = np.append(x2_val_MC_folder, x2_val_MC_time_folder[i])
                else:

                    print(' Calculating time instant '+str(i)+' from '+case.split('/')[-1]+', '+folders_list[n].split('/')[-1]) 
                    for j in range(n_ligaments):
                    
                        file_name = prefix + str(j) + '.'+ str(i) +'.csv'
                        file_path = path+'/'+file_name
                    
                        # Check if file is empty
                        try:
                            df = pd.read_csv(file_path) 
                        except:
                            #print('File ' + file_name + ' is empty')
                            continue
                    
                        for k in range(len(df['Points:0'])):
                            x1_val_folder = np.append(x1_val_folder,df['Points:0'][k]*1e3)
                            x2_val_folder = np.append(x2_val_folder,df[coord_x2][k]*1e3) 
                            x1_val_time_folder[i].append(df['Points:0'][k]*1e3)
                            x2_val_time_folder[i].append(df[coord_x2][k]*1e3)
                        
                        
                            if 'MAIN_COLOR' in df.columns:
                                if df['MAIN_COLOR'][k] == 1:
                                    x1_val_MC_folder  = np.append(x1_val_MC_folder,df['Points:0'][k]*1e3)
                                    x2_val_MC_folder  = np.append(x2_val_MC_folder,df[coord_x2][k]*1e3) 
                                    x1_val_MC_time_folder[i].append(df['Points:0'][k]*1e3)
                                    x2_val_MC_time_folder[i].append(df[coord_x2][k]*1e3)
                        
                    x1_val_time_folder[i], x2_val_time_folder[i] = range_vector_pairs(x1_val_time_folder[i], x2_val_time_folder[i])
                    data_trajectory_time_i = [x1_val_time_folder[i], x1_val_MC_time_folder[i],
                                              x2_val_time_folder[i], x2_val_MC_time_folder[i]]

                    pickleSave(data_trajectory_time_i, file_data_time_i)
                
            data_trajectory_folder = [x1_val_folder, x1_val_time_folder,
                                    x1_val_MC_folder, x1_val_MC_time_folder,
                                    x2_val_folder, x2_val_time_folder,
                                    x2_val_MC_folder, x2_val_MC_time_folder]
            
            pickleSave(data_trajectory_folder, file_data_trajectory_folder)
                   
        x1_val         = np.append(x1_val, x1_val_folder)
        x1_val_time    = np.append(x1_val_time, x1_val_time_folder)
        x1_val_MC      = np.append(x1_val_MC, x1_val_MC_folder)
        x1_val_MC_time = np.append(x1_val_MC_time, x1_val_MC_time_folder)
        x2_val         = np.append(x2_val, x2_val_folder)
        x2_val_time    = np.append(x2_val_time, x2_val_time_folder)
        x2_val_MC      = np.append(x2_val_MC, x2_val_MC_folder)
        x2_val_MC_time = np.append(x2_val_MC_time, x2_val_MC_time_folder)
    

    x1_val   , x2_val    = range_vector_pairs(x1_val, x2_val)
    x1_val_MC, x2_val_MC = range_vector_pairs(x1_val_MC, x2_val_MC)
    
    x1_valD = x1_val/D ; x1_valD_MC = x1_val_MC/D
    x2_valD = x2_val/D ; x2_valD_MC = x2_val_MC/D
    
    # Check what does x_val_MC_time contains
    numerical_correlation = [x1_val   , x2_val   , x1_valD   , x2_valD, 
                             x1_val_MC, x2_val_MC, x1_valD_MC, x2_valD_MC,
                             x1_val_time, x2_val_time, x1_val_MC_time, x2_val_MC_time]
    
    return numerical_correlation
    
# 
# END FUNCTION process_raw_data
# --------------------------------------- 
    



# ---------------------------------------
# FUNCTION range_vector_pairs
#    
def range_vector_pairs(a, b):
    """
    Range pairs of vectors a, b by increasing values of a
    """
    assert (len(a) == len(b)), 'Vectors do not have same length !'
    
    a = np.array(a)        ; b = np.array(b)
    a = a.reshape(len(a),1); b = b.reshape(len(b),1)
    values = np.concatenate((a,b), axis = 1)
    values = np.ndarray.tolist(values)
    values  = sorted(values, key=itemgetter(0))
    values  = np.array(values)
    a_range = np.zeros(len(values)); b_range = np.zeros(len(values))
    for j in range(len(values)):
        a_range[j] = values[j][0]
        b_range[j] = values[j][1]
    
    return a_range, b_range
# 
# END FUNCTION range_vector_pairs 
# --------------------------------------- 


# Create class trajectory_vertical
# --------------------------------------- 
# 
class trajectory_lateral():
    def __init__(self, x_traj, D, q):
        self.x       = x_traj
        self.xD      = x_traj/D
        
        
        # For the moment, the only available trajectory is the one by Becker
        self.name  = 'becker'
        self.std   = 0.54
        yTr = lambda xD: 2.32*q**0.09*xD**0.32
        
        # Top branch (y>0)
        self.yD_mean_top   = np.array([yTr(xD) for xD in self.xD])
        self.yD_upper_top  = self.yD_mean_top + self.std
        self.yD_lower_top  = self.yD_mean_top - self.std
        # Bottom branch (y<0)
        self.yD_mean_bottom  = -1*self.yD_mean_top
        self.yD_upper_bottom = self.yD_mean_bottom + self.std
        self.yD_lower_bottom = self.yD_mean_bottom - self.std
        self.plot_limits  = [[-D/2, self.x[-1]],
                             [-1*D*yTr(self.xD[-1])*1.1, D*yTr(self.xD[-1])*1.1]]
        self.plotD_limits = []
        for i in range(2):
            self.plotD_limits.append([x/D for x in self.plot_limits[i]])
        

# 
# END CLASS trajectory_lateral
# --------------------------------------- 
            


# Create class trajectory_vertical
# --------------------------------------- 
# 
class trajectory_vertical():
    def __init__(self, x_traj, D):
        self.x       = x_traj
        self.xD      = x_traj/D
        self.z_mean  = []
        self.z_upper = []
        self.z_lower = []
        self.name  = ''
        self.std   = np.nan

    def get_trajectory(self, D, q, Weaero = [], correlation = 'becker'):
    
        if correlation == 'becker':
            zTr  = lambda xD : 1.57*q**0.36*np.log(1+3.81*xD)
            self.std  = 0.81
            self.name = 'Becker'
        elif correlation == 'becker_2':
            zTr  = lambda xD : 1.48*q**0.42*np.log(1+3.56*xD)
            self.name    = 'Becker'
        elif correlation == 'ragucci':
                assert Weaero, 'Ragucci correlation needs input Weaero'
                zTr  = lambda xD : 2.698*q**0.441*Weaero**(-0.069)*xD**0.367
                self.name     = 'Ragucci'
        
        self.zD_mean = np.array([zTr(xD) for xD in self.xD])
        self.z_mean  = D*self.zD_mean
        self.plot_limits  = [[-D/2, self.x[-1]],[0, D*zTr(self.xD[-1])*1.1]]
        self.plotD_limits = []
        for i in range(2):
            self.plotD_limits.append([x/D for x in self.plot_limits[i]])
        if self.std:
            self.zD_upper = self.zD_mean+self.std
            self.z_upper  = D*self.zD_upper
            self.zD_lower = self.zD_mean-self.std
            self.z_lower  = D*self.zD_lower
# 
# END CLASS trajectory_vertical
# --------------------------------------- 
            
            





