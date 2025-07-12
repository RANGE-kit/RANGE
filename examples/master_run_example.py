# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_py.ga_abc import GA_ABC
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import energy_computation
from RANGE_py.input_output import save_energy_summary

import numpy as np
import matplotlib.pyplot as plt
import os

#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#from ase.io import read, write

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
from xtb.ase.calculator import XTB
#from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
water = 'xyz_structures/Water.xyz'
methane = 'xyz_structures/methane.xyz'
organic_sub = 'xyz_structures/Substrate.xyz'
single_atom = 'xyz_structures/Single_atom.xyz'
slab_surface = 'xyz_structures/Slab_BaTiO3_7layer.xyz'
copper_12 = 'xyz_structures/Cu12.xyz'
co2 = 'xyz_structures/CO2.xyz'
#O_atoms_idx_in_slab = tuple([at.index for at in read(slab_surface) if at.symbol=='O'])

input_molecules = [water]#, co2]
input_num_of_molecules = [5]#, 1]
 
'''
at_position : X,Y,Z, (Euler_X, Euler_Y, Euler_Z)
in_box : xlo,ylo,zlo,xhi,yhi,zhi 
in_sphere_shell : X,Y,Z, R_x, R_y, R_z
'on_surface' : id_substrate, (lo, hi) of adsorption distance, id_atom_surf, id_atom_orientation. e.g. (0,(1.9, 2.1),0,1),
replace: id_substrate, tuple/list of index of available atoms
'''
input_constraint_type = [#'at_position', 
                         #'on_surface',
                         'in_sphere_shell'
                         ]
input_constraint_value = [#(0,0,0,0,0,0), 
                          #(0,(1.9, 2.1),1,0),
                          (0,0,0,4,4,4)
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
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = None, # None = single point calc, 
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_eps = None, 
                                 coarse_calc_sig = None, 
                                 coarse_calc_chg = [0.417, -0.834, 0.417]*5 , 
                                 coarse_calc_step = 10, 
                                 coarse_calc_fmax = 10,
                                 )

# Put together and run the algorithm
output_folder_name = 'results-1'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=20, max_iteration=10, 
                      ga_interval=2, ga_parents=3, mutate_rate=0.2, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      restart_from_pool = 'structure_pool.db',
                      )
all_x, all_y, all_name = optimization.run(print_interval=1)

print( "Step 4: See results: use analysis script" )
