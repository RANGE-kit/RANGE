# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from ga_abc import GA_ABC
from cluster_model import cluster_model
from energy_calculation import energy_computation
from utility import save_energy_summary

import numpy as np
import matplotlib.pyplot as plt
import os

#from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.io import read, write

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
from xtb.ase.calculator import XTB
#from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
water = 'Water.xyz'
methane = 'methane.xyz'
organic_sub = 'Substrate.xyz'
single_atom = 'Single_atom.xyz'
slab_surface = 'Slab_BaTiO3_7layer.xyz'
copper_12 = 'Cu12.xyz'
co2 = 'CO2.xyz'

O_atoms_idx_in_slab = tuple([at.index for at in read(slab_surface) if at.symbol=='O'])

input_molecules = [copper_12, co2]
input_num_of_molecules = [1, 1]
 
'''
at_position : X,Y,Z, (Euler_X, Euler_Y, Euler_Z)
in_box : xlo,ylo,zlo,xhi,yhi,zhi 
in_sphere_shell : X,Y,Z, R_x, R_y, R_z
'on_surface' : id_substrate, (lo, hi) of adsorption distance, id_atom_surf, id_atom_orientation. e.g. (0,(1.9, 2.1),0,1),
replace: id_substrate, tuple/list of index of available atoms
'''
input_constraint_type = ['at_position', 
                         'on_surface', 
                         ]
input_constraint_value = [(0,0,0,0,0,0), 
                          (0,(1.9, 2.1),1,0),
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
ase_calculator = XTB(method="GFN2-xTB") #TBLite(method="GFN2-xTB", verbosity=-1)
#ase_calculator = mace_mp(model='small', dispersion=False, default_dtype="float64", device='cuda')

geo_opt_parameter = dict(fmax=0.2, steps=20)

# for ASE
"""
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = None, # None = single point calc, 
                                 )
"""
# for external
calculator_command_line = 'xtb  {input_xyz}  --silent'
geo_opt_control_line = dict(method='xtb-gfn2', run_type='single_point')

computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = calculator_command_line,
                                 calculator_type = 'external', 
                                 geo_opt_para = geo_opt_control_line 
                                 )

print( "Step 3: Run" )
# Put together and run the algorithm
output_folder_name = 'results'

optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=20, max_iteration=5, 
                      ga_interval=20, ga_parents=5, mutate_rate=0.2, mutat_sigma=0.05,
                      output_directory = output_folder_name
                      )
best_x, best_y, all_x, all_y, all_name = optimization.run(print_interval=1)


print( "Step 4: See results" )
# Emphasize the best structure
best_structure = computation.vector_to_cluster(best_x)
#all_structures = [ computation.vector_to_cluster(x) for x in all_x ]
#view(best_structure)
write( os.path.join(output_folder_name,'best.xyz'), best_structure)

# See and save the energy as function of appearance
ydat = np.round(all_y, 6)
xdat = np.arange( len(ydat) )
#mask = all_y < 1e6
#xdat, ydat = xdat[mask], all_y[mask]
fig, axs = plt.subplots(1,2,figsize=(12,6),tight_layout=True)
axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)
#plot_atoms(best_structure, axs[1], radii=0.3, rotation=('0x,0y,0z'))

# Rank and save results
sorted_idx = np.argsort(all_y)[::-1]
ydat = np.round(all_y[sorted_idx], 6)
save_energy_summary(sorted_idx, ydat, all_name, os.path.join(output_folder_name, 'energy_summary.log') )

xdat = np.arange( len(ydat) )
axs[1].plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
axs[1].tick_params(direction='in',labelsize=8)

plt.savefig( os.path.join(output_folder_name,"my_plot.png"), dpi=200)
plt.show()

