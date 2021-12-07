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



def BlasiusBLTurb(x, u0, kinVisc):
     
    delta = 0.37*x**(4/5)/((u0/kinVisc)**(1/5))
    return delta

def getfR2(b, deltaInlet, Rad):
    fR2 = 1 - 1/3*((b - deltaInlet)/Rad)**2 
    #fR2 = (3*Rad**2 - b**2 + 2*b*deltaInlet - deltaInlet**2)/(3*Rad**2)
    return fR2

def integrate(xvalues,fvalues):
    
    F = 0
    for i in range(len(fvalues)-1):
        F += (xvalues[i+1]-xvalues[i])*(fvalues[i+1]+fvalues[i])/2
    return F

def solveBL(R, uG, uTau, b, kinViscG, delta, zMin):
    

    Q_viscousSublayer = uTau**2*zMin**2 /(2*dynViscG)
    Q_logLayer        = uTau/kappa*( delta*np.log(delta*uTau/kinViscG) + 
                                      (delta - zMin)*(kappa*cPlus - 1) - 
                                      zMin*np.log(zMin*uTau/kinViscG) )
    uMaxNom = uTau*(1/kappa*np.log(delta*uTau/kinViscG) + cPlus) 
    uMaxDen = 1 - ((b - delta)/R)**2
    fR2     = (b**2 + (b - delta)**2/3 - b*(b - delta))/(R**2)
    Q_outerLayer      = uMaxNom/uMaxDen*(b - delta)*(1 - fR2)
    
    QTot_RHS = Q_viscousSublayer + Q_logLayer + Q_outerLayer
    
    return QTot_RHS
                                  


    


def velocityProfile3D(yGrid, zGrid, by, bz, delta, uC):
    
    uPoints = np.zeros( (len(zGrid),len(zGrid[0])) )
    
    for i in range(len(zGrid)):
        for j in range(len(zGrid[0])):
            y = yGrid[i][j]
            z = zGrid[i][j]
            if y >= (by - delta ) and z <= delta:     
                # Region 1
                u = uC * ( ( by - y )/delta )**(1/7) * ( z/delta )**(1/7)
            elif y > (by - delta ) and z > delta:    
                # Region 2
                u = uC * ( ( by - y )/delta )**(1/7) 
            elif y < (by - delta ) and z < delta:
                # Region 3
                u = uC * ( z/delta )**(1/7) 
            elif y < (by - delta ) and z > delta:
                # Region 4
                u = uC
            uPoints[i,j] = u
           
    return uPoints

##########################################################
####                   INPUT                          ####
##########################################################

#------- Operating conditions ---------
q  = 6
uG = 100
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



              
# Jetter un coup d'oeil:
   # https://en.wikiversity.org/wiki/Fluid_Mechanics_for_Mechanical_Engineers/Internal_Flows

##########################################################
####        Flat plate boundary layer                 ####
##########################################################

x          = Symbol('x')
xInjBL     = solve( BlasiusBLTurb(x, uG, kinViscG) - 5e-3, x)
xInjBL     = xInjBL[0]
xRelInlet  = xInjBL - 120e-3
RexInlet   = xRelInlet*uG/kinViscG
deltaInlet = BlasiusBLTurb(xRelInlet, uG, kinViscG)
deltaInlet = float(deltaInlet)


zPoints   = np.linspace(0, b, 20000)
                               
                               
##########################################################
####           Operating conditions                   ####
##########################################################

uL     = uG*np.sqrt(q*rhoG/rhoL)
u0     = uL*(DInj/D0)**2
ReL    = DInj*uL/kinViscL
ReG    = Dh*uG/kinViscG
WeL    = rhoL*uL**2*DInj/surfTens
WeG    = rhoG*uG**2*DInj/surfTens
WeRel  = rhoG*(uG - uL)**2*DInj/surfTens
WeAero = rhoG*uL**2*DInj/surfTens
                                     

    
ReL_Inlet = D0*u0/kinViscL
ReG_Inlet = Dh*uG/kinViscG

Oh = dynViscL/np.sqrt(rhoL*surfTens*DInj)

massFlowInjector = rhoL*np.pi/4*DInj**2*uL

##########################################################
####           Characteristic times  [ms]             ####
##########################################################
t_ab  = np.sqrt(rhoL/rhoG)*DInj/(uG-uL)*1e6
t_nb  = DInj/uL*WeL**(1/3)*1e6
t_in  = DInj/uL*1e6
t_cap = np.sqrt(rhoL*DInj**3/surfTens)*1e6
t_visc_1 = DInj**2/kinViscL*1e6
t_visc_2 = dynViscL*DInj/surfTens*1e6


#------- Print information to screen ---------

print('\n---------- OPERATING CONDITIONS  ---------- ')
print(f'             q  = {q}')
print(f'             uG = {uG} m/s')
print(f'             uL = {uL:.3f} m/s')
print(f'            ReL = {ReL:.3f} ')
print(f'            ReG = {ReG:.3f} ')
print(f'            WeL = {WeL:.3f} ')
print(f'            WeG = {WeG:.3f} ')
print(f'          WeRel = {WeRel:.3f} ')
print(f'         Weaero = {WeAero:.3f} ')
print(f'      ReL_Inlet = {ReL_Inlet:.3f} ')
print(f'             Oh = {Oh:.3f} ')
print(f'             uL = {uL:.3f} m/s')
print('\n---------- CHARACTERISTIC TIMES  [µs] ---------- ')
print(f'           t_ab = {t_ab:.3f} ')
print(f'           t_nb = {t_nb:.3f} ')
print(f'            T_b = {t_ab/t_nb:.3f} ')
print(f'           t_in = {t_in:.3f} ')
print(f'          t_cap = {t_cap:.3f} ')
print(f'       t_visc_1 = {t_visc_1:.3f} ')
print(f'       t_visc_2 = {t_visc_2:.3f} ')
print('\n---------- BOUNDARY CONDITIONS ---------- ')
print(f'             u0 = {u0:.3f} m/s')
print(f'          delta = {deltaInlet*1e3:.3f} mm')
print('\n-------- For LAGRANGIAN injection ------- ')
print(f'           mdot = {massFlowInjector:.3f}\n')
print('-----------------------------------------\n')
                                                          
                               




##########################################################
####        Profile Calculations  (1/nth law)         ####
##########################################################

n = 7


bz    = b
by    = wChannel/2
delta = deltaInlet

f = n/(n+1)

# Get uC
TOCHACO = (f*delta)**2 + f*delta*(by+bz-2*delta) + (bz-delta)*(by-delta)
uC = uG* by * bz / TOCHACO

# Get flow rates to check
Q1 = uC*(f*delta)**2
Q2 = uC*f*delta*(bz - delta)
Q3 = uC*f*delta*(by - delta)
Q4 = uC*(by - delta)*(bz - delta)

# Get grid and velocity profile
yPoints = np.linspace(0, by, 1000)
zPoints = np.linspace(0, bz, 1000)

yGrid, zGrid = np.meshgrid(yPoints, zPoints)
uProfile = velocityProfile3D(yGrid, zGrid, by, bz, delta, uC)



print('\n---------- 3D, 1/nth BL profile ---------- ')
print(f'             bz = {bz*1e3} mm    ')
print(f'             by = {by*1e3} mm    ')
print(f'             u0 = {u0:.3f} m/s')
print(f'             uC = {uC:.6f} m/s  ')
print(f'          delta = {deltaInlet*1e3:.3f} mm')

# Plot 2D profile
yPoints = np.linspace(0,bz,10000)
uPoints = np.ones(len(yPoints))*uC
for i in range(len(yPoints)):
    if yPoints[i] <= delta:
        uPoints[i] = (yPoints[i]/delta)**(1/n)*uC


plt.figure(figsize=(fPic*18,fPic*13))
plt.title(f"One-{n}th power law, flat outer layer")
plt.plot(yPoints*1e3, uPoints,'k',linewidth=7)
plt.xlabel(r'y [mm]')
plt.xlim(0,5)
plt.ylabel(r'u [m/s]')
plt.grid()
plt.show()

#%%
folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/appendices_figures/'

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# Plot 3D profile
fig = plt.figure(figsize=(fPic*20,fPic*13))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(yGrid*1e3, zGrid*1e3, uProfile, vmin = 0, vmax = 110,
                cmap=cm.inferno, edgecolor='none')
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
ax.set_yticks([0,5, 10, 15,20])
ax.set_zlabel('u [m/s]')
ax.set_zbound([0,110])
#ax.set_title(f'Velocity profile in 3D, 1/{n}th law')
plt.tight_layout()
#fig.colorbar(surf, shrink=0.5, aspect=10)
#plt.savefig(folder_manuscript+'gaseous_inlet_u_profile_'+str(uG)+'.eps')
plt.show()

plt.rcParams['text.usetex'] = False


#%%
# Standalone colormap
#import pylab as pl
plt.rcParams['text.usetex'] = True

a = np.array([[0,110]])
plt.figure(figsize=(fPic*18, fPic*1.5))
#plt.figure(figsize=(fPic*1.5, fPic*18))
img = plt.imshow(a, cmap="inferno")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation="horizontal", cax=cax)
cbar.set_label('u [m/s]')
plt.tight_layout()
#plt.savefig(folder_manuscript+'inlet_colorbar.eps',bbox_inches="tight")

plt.rcParams['text.usetex'] = False