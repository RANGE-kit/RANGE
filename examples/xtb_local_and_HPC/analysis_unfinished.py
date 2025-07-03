# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 08:30:28 2025

@author: d2j
"""

import numpy as np
import os
import matplotlib.pyplot as plt


output_folder_name = 'results-inbox'
jobs_name = [ f for f in os.listdir(output_folder_name) if 'compute_' in f ]

jobs_energy = []
for j in jobs_name:
    log = os.path.join( output_folder_name , j , 'job.log')
    with open(log,'r') as f1:
        energy = [line.split() for line in f1.readlines() if "TOTAL ENERGY" in line ]
        energy1 = [ line.split() for line in f1.readlines() if "total energy " in line ]
    if len(energy)>0:
        energy = float(energy[-1][3]) # last energy. Value is the 4th item
    elif len(energy1)>0:
        energy = float(energy1[-1][4])
    else:
        energy = 1e7
    jobs_energy.append( energy )
        
jobs_energy = np.array(jobs_energy)
sorted_idx = np.argsort(jobs_energy)[::-1]
for i in sorted_idx[-10:]:
    print( jobs_energy[i], jobs_name[i] )
        
ydat = jobs_energy[sorted_idx]
xdat = np.arange( len(ydat) )  
mask = ydat < 1e6
xdat, ydat = xdat[mask], ydat[mask]

fig, axs = plt.subplots(1,2,figsize=(8,3),tight_layout=True, dpi=200)
axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)

axs[1].plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
axs[1].tick_params(direction='in',labelsize=8)

#plt.savefig( os.path.join(output_folder_name,"my_plot.png"), dpi=200)
plt.show()
