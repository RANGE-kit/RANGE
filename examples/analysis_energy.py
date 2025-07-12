# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 08:30:28 2025

@author: d2j
"""

import numpy as np
import matplotlib.pyplot as plt
from RANGE_py.input_output import save_energy_summary #, save_best_structure


results = save_energy_summary(output_file='energy_summary.log', db_path='structure_pool.db', directory_path='results')

# Plot energy vs appearance and ranked energies
fig, axs = plt.subplots(1,2,figsize=(8,3),tight_layout=True)
# From bee type to color
color_map = {'EM':'r', 'OL':'g', 'SC':'b', 'GA':'c'}

mask = results['unranked_ener'] < 1e6
ydat = results['unranked_ener'][mask]
xdat = results['appear_id'][mask]
op_type = results['op_type'][mask]
#axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
colors = [ color_map[b] for b in op_type ]
axs[0].scatter(xdat, ydat, marker='o', s=8, c=colors, alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)

mask = results['ranked_ener'] < 1e6
ydat = results['ranked_ener'][mask]
xdat = results['ranked_id'][mask]
op_type = results['op_type'][mask]
for k,v in color_map.items():
    mask = op_type==k
    axs[1].scatter(xdat[mask], ydat[mask], marker='o', s=8, c=v, label=k,alpha=0.8)
    axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
    axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
    axs[1].tick_params(direction='in',labelsize=8)

plt.savefig( "my_plot.png", dpi=200)
#plt.show()

#save_best_structure(output_folder_name, all_name[sorted_idx[0]] , os.path.join(output_folder_name, 'best.xyz'))
#print( 'Best structure: ', all_name[sorted_idx[0]] )
