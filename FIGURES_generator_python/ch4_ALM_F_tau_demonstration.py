# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:43:10 2021

@author: d601630
"""


FFIG = 0.5
import matplotlib.pyplot as plt
import numpy as np

folder_manuscript='C:/Users/d601630/Documents/GitHub/Thesis_Carlos/part2_developments/figures_ch4_SLI/'


plt.rcParams['xtick.labelsize'] = 60*FFIG #40*FFIG
plt.rcParams['ytick.labelsize'] = 60*FFIG#40*FFIG
plt.rcParams['axes.labelsize']  = 60*FFIG #40*FFIG
plt.rcParams['axes.labelpad']   = 30*FFIG
plt.rcParams['axes.titlesize']  = 50*FFIG
plt.rcParams['legend.fontsize'] = 40*FFIG  #30*FFIG
plt.rcParams['lines.linewidth'] = 6*FFIG
plt.rcParams['legend.loc']      = 'lower right'
plt.rcParams['legend.framealpha']      = 1.0
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

rho = 7.21
mu  = 1.8162E-5


dp = 1e4


Rex = np.linspace(1e3,1e5,100000)


uG75  = 75
rf_uG75   = dp/(0.185*rho*uG75**2*np.log10(Rex)**(-2.584))

uG100 = 100
rf_uG100   = dp/(0.185*rho*uG100**2*np.log10(Rex)**(-2.584))


plt.figure(figsize=(FFIG*18,FFIG*13))
plt.semilogx(Rex,rf_uG75,'k',label='u$_g$ = 75 m s$^{-1}$')
plt.semilogx(Rex,rf_uG100,'b',label='u$_g$ = 100 m s$^{-1}$')
plt.xlabel('Re$_x$')
plt.ylabel('r$_F$')
plt.ylim([10,90])
plt.yticks([10,30,50,70,90])
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig(folder_manuscript+'ALM_rF_vs_Rex.pdf')
plt.savefig(folder_manuscript+'ALM_rF_vs_Rex.eps',format='eps',dpi=1000)
plt.show()
plt.close()