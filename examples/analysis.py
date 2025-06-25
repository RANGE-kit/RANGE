# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 08:30:28 2025

@author: d2j
"""

import numpy as np
import os
import matplotlib.pyplot as plt


with open('energy_summary.log', 'r') as f1_input:    
    lines = [ line.replace('|','').split() for line in f1_input.readlines()  ]
    col_name = lines[0]    
    rank_idx, appear_idx, energy, compute_id = [],[],[],[]
    for line in lines[1:]:
        rank_idx.append( int(line[0]) ) 
        appear_idx.append( int(line[1]) )
        energy.append( float(line[2]) )
        compute_id.append( line[3] )
        
energy = np.array(energy)

fig, axs = plt.subplots(1,1, figsize=(4,3),tight_layout=True, dpi=200)

sorted_idx = np.argsort( energy )[::-1]
ydat = energy[sorted_idx]
xdat = np.arange( len(ydat) )  
axs.plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs.set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs.set_ylabel('Energy',fontsize=10) ## input Y name
axs.tick_params(direction='in',labelsize=8)

#plt.savefig("my_plot.png", dpi=200)
plt.show()