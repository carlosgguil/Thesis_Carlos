# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:09:33 2019

@author: d601630
"""

from sympy.solvers import solve
from sympy import Symbol
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
fPic  = 0.5
kappa = 0.4
cPlus = 5.0

# rcParams for plots
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['axes.titlesize'] = 30

plt.rcParams['xtick.labelsize'] = 60*fPic #40*fPic
plt.rcParams['ytick.labelsize'] = 60*fPic#40*fPic
plt.rcParams['axes.labelsize']  = 60*fPic #40*fPic
plt.rcParams['axes.labelpad']   = 50*fPic
plt.rcParams['axes.titlesize']  = 50*fPic
plt.rcParams['legend.fontsize'] = 40*fPic  #30*fPic
plt.rcParams['lines.linewidth'] = 2*fPic
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0


##########################################################
####                   INPUT                          ####
##########################################################

#------- Operating conditions ---------
q  = 6
uG100 = 100
uG75 = 75
#------- Air properties ---------
rhoG     = 7.21
dynViscG = 1.8162e-5
kinViscG = dynViscG/rhoG

#------- Liquid (kerosene) properties ---------
rhoL     = 795
dynViscL = 1.5e-3
kinViscL = dynViscL/rhoL
surfTens = 22e-3


#------- Fuel injector nozzle geometry ---------
D0   = 1.5e-3
DInj = 0.45e-3

#------- Channel geometry ---------
hChannel = 40e-3
b        = hChannel/2
wChannel = 25e-3
Dh       = 2*hChannel*wChannel/(hChannel + wChannel) #Hydraulic diameter



              

                               
##########################################################
####           Operating conditions  UG75                 ####
##########################################################



Re_uG75  = Dh*uG75/kinViscG
Re_uG100 = Dh*uG100/kinViscG
    


# Large eddies
L = Dh/2
L_uG75 = L
L_uG100 = L
fL_uG75  = 0.4*uG75/Dh*Re_uG75**(-1/8)
fL_uG100 = 0.4*uG100/Dh*Re_uG100**(-1/8)
tL_UG75  = 1/fL_uG75
tL_UG100 = 1/fL_uG100

# Energy containing eddies
le_UG75 = 0.05*Dh*Re_uG75**(-1/8)
le_UG100 = 0.05*Dh*Re_uG100**(-1/8)
fe_uG75  = 4*uG75/Dh
fe_uG100 = 4*uG100/Dh
te_UG75  = 1/fe_uG75
te_UG100 = 1/fe_uG100

# Most dissipative eddies
ld_UG75 = 20*Dh*Re_uG75**(-0.78)
ld_UG100 = 20*Dh*Re_uG100**(-0.78)
fd_uG75  = 0.02*uG75/Dh*Re_uG75**(0.56)
fd_uG100  = 0.02*uG100/Dh*Re_uG100**(0.56)
td_UG75  = 1/fd_uG75
td_UG100 = 1/fd_uG100

# Kolmogorov eddies
lk_UG75 = 4*Dh*Re_uG75**(-0.78)
lk_UG100 = 4*Dh*Re_uG100**(-0.78)
fk_uG75  = 0.06*uG75/Dh*Re_uG75**(0.56)
fk_uG100  = 0.06*uG100/Dh*Re_uG100**(0.56)
tk_UG75  = 1/fk_uG75
tk_UG100 = 1/fk_uG100

# Kolmogorov eddies (piomelli)
tau_L_uG75 = Dh/2/uG75
f_L_uG75 = 1/tau_L_uG75/1e3
eta_uG75 = L*Re_uG75**(-0.75)
tau_eta_uG75 = tau_L_uG75*Re_uG100**(-0.5)*1e3
f_eta_uG75 = 1/tau_eta_uG75
tau_L_uG100 = Dh/2/uG100
f_L_uG7100 = 1/tau_L_uG100/1e3
eta_uG100 = L*Re_uG100**(-0.75)
tau_eta_uG100 = tau_L_uG100*Re_uG100**(-0.5)*1e3
f_eta_uG7100 = 1/tau_eta_uG100