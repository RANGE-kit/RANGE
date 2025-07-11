# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_py.ga_abc import GA_ABC
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import energy_computation, RigidLJQ_calculator
from RANGE_py.input_output import save_energy_summary, save_best_structure

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
carbon = '../xyz_structures/C.xyz'

input_molecules = [carbon]
input_num_of_molecules = [60]
 
input_constraint_type = ['in_box']
input_constraint_value = [(0,0,0, 11,11,11, 4,4,4, 7,7,7) ]

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
# for ASE
ase_calculator = XTB(method="GFN2-xTB") 
geo_opt_parameter = dict(fmax=0.2, steps=20)
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = None, # None = single point calc, 
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_eps = None, 
                                 coarse_calc_sig = None, 
                                 coarse_calc_chg = None , 
                                 coarse_calc_step = 10, 
                                 coarse_calc_fmax = 10,
                                 )

# for external xTB
xtb_exe_path = '/Users/d2j/Downloads/xtb-6.7.1/bin/xtb.exe'
calculator_command_line = xtb_exe_path + " --gfn2  {input_xyz} --etemp 2500 --opt normal --cycles 300 --iterations 1000 "
geo_opt_control_line = dict(method='xTB')
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = calculator_command_line,
                                 calculator_type = 'external', 
                                 geo_opt_para = geo_opt_control_line ,
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_eps = None, 
                                 coarse_calc_sig = None, 
                                 coarse_calc_chg = None , 
                                 coarse_calc_step = 10, 
                                 coarse_calc_fmax = 10,
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=20, max_iteration=10, 
                      ga_interval=2, ga_parents=3, mutate_rate=0.2, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      )
all_x, all_y, all_name = optimization.run(print_interval=1)

print( "Step 4: See results" )
# Save ranked energies and their appearance
sorted_idx = np.argsort(all_y)
save_energy_summary(sorted_idx, all_y, all_name, os.path.join(output_folder_name, 'energy_summary.log') )
save_best_structure(output_folder_name, all_name[sorted_idx[0]] , os.path.join(output_folder_name, 'best.xyz'))
print( 'Best structure: ', all_name[sorted_idx[0]] )

# Plot energy vs appearance and ranked energies
#ydat = np.round(all_y, 6)
xdat = np.arange( len(all_y) )
mask = all_y < 1e6
xdat, ydat = xdat[mask], all_y[mask]
fig, axs = plt.subplots(1,2,figsize=(12,6),tight_layout=True)
axs[0].plot(xdat, ydat, marker='o', ms=4, lw=2, color='orange', label='All data',alpha=0.8)
axs[0].set_xlabel('Structures (appearance)',fontsize=10) ## input X name
axs[0].set_ylabel('Energy',fontsize=10) ## input Y name
axs[0].tick_params(direction='in',labelsize=8)

ydat = all_y[sorted_idx]
xdat = np.arange( len(all_y) )
mask = ydat < 1e6
xdat, ydat = xdat[mask], ydat[mask]
axs[1].plot(xdat, ydat, marker='o', ms=4, lw=2, color='skyblue', label='All data',alpha=0.8)
axs[1].set_xlabel('Structures (re-ordered)',fontsize=10) ## input X name
axs[1].set_ylabel('Energy',fontsize=10) ## input Y name
axs[1].tick_params(direction='in',labelsize=8)
plt.savefig( os.path.join(output_folder_name,"my_plot.png"), dpi=200)
#plt.show()
