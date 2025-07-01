# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_py.ga_abc import GA_ABC
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import energy_computation
from RANGE_py.utility import save_energy_summary, save_best_structure

import numpy as np
import matplotlib.pyplot as plt
import os

#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#from ase.io import read, write

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
#from xtb.ase.calculator import XTB
#from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
carbon = '../xyz_structures/C.xyz'

input_molecules = [carbon]
input_num_of_molecules = [60]
 
'''
at_position : X,Y,Z, (Euler_X, Euler_Y, Euler_Z)
in_box : xlo,ylo,zlo,xhi,yhi,zhi 
in_sphere_shell : X,Y,Z, R_x, R_y, R_z
'on_surface' : id_substrate, (lo, hi) of adsorption distance, id_atom_surf, id_atom_orientation. e.g. (0,(1.9, 2.1),0,1),
replace: id_substrate, tuple/list of index of available atoms
'''
input_constraint_type = ['in_box', 
                         ]
input_constraint_value = [ (0,0,0,11,11,11, 4,4,4,7,7,7)  #(0,0,0,7,7,7, 1.5,1.5,1.5,5.5,5.5,5.5), 
                          ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        #pbc_box=(22.90076, 23.00272, 31.95000),
                        )
cluster.init_molelcules()
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Set the way to compute energy
calculator_command_line = " ~/bin/xtb  {input_xyz} --etemp 2500 --opt tight  --cycles 500 --iterations 1000 "

geo_opt_control_line = dict(method='xtb-gfn2')

computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = calculator_command_line,
                                 calculator_type = 'external', 
                                 geo_opt_para = geo_opt_control_line 
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=20, max_iteration=3, 
                      ga_interval=2, ga_parents=3, mutate_rate=0.2, mutat_sigma=0.05,
                      output_directory = output_folder_name
                      )
all_x, all_y, all_name = optimization.run(print_interval=1)


print( "Step 4: See results" )
# Save ranked energies and their appearance
sorted_idx = np.argsort(all_y)
save_energy_summary(sorted_idx, all_y, all_name, os.path.join(output_folder_name, 'energy_summary.log') )
save_best_structure(output_folder_name, all_name[sorted_idx[0]] , os.path.join(output_folder_name, 'best.xyz'))
print( 'Best structure: ', all_name[sorted_idx[0]] )

# Plot energy vs appearance and ranked energies
ydat = np.round(all_y, 6)
xdat = np.arange( len(ydat) )
#mask = all_y < 1e6
#xdat, ydat = xdat[mask], all_y[mask]
fig, axs = plt.subplots(1,2,figsize=(12,6),tight_layout=True)
axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)
ydat = ydat[sorted_idx]
axs[1].plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
axs[1].tick_params(direction='in',labelsize=8)
plt.savefig( os.path.join(output_folder_name,"my_plot.png"), dpi=200)
#plt.show()
