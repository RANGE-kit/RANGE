# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 08:30:28 2025

@author: d2j
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from RANGE_py.input_output import read_structure_from_db, read_structure_from_directory, save_energy_summary, save_best_structure


# Method 1: use .db
db_path = 'structure_pool.db'
vec, energy, name = read_structure_from_db( db_path, 'all', None )

# Method 2: use directory
directory_path = 'results'
vec, energy, name = read_structure_from_directory( directory_path, 'all', None )

# Sort energy
sorted_idx = np.argsort(energy)[::-1]
save_energy_summary(sorted_idx, energy, name, 'energy_summary.log') 
#save_best_structure(output_folder_name, all_name[sorted_idx[0]] , os.path.join(output_folder_name, 'best.xyz'))
#print( 'Best structure: ', all_name[sorted_idx[0]] )

# Plot energy vs appearance and ranked energies
#ydat = np.round(all_y, 6)
xdat = np.arange( len(energy) )
mask = energy < 1e6
xdat, ydat = xdat[mask], energy[mask]
fig, axs = plt.subplots(1,2,figsize=(12,6),tight_layout=True)
axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)

ydat = energy[sorted_idx]
xdat = np.arange( len(energy) )
mask = ydat < 1e6
xdat, ydat = xdat[mask], ydat[mask]
axs[1].plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
axs[1].tick_params(direction='in',labelsize=8)
plt.savefig( "my_plot.png", dpi=200)
#plt.show()
