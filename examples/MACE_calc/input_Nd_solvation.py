# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
#import matplotlib.pyplot as plt
import os

#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths
from ase import Atoms

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
from xtb.ase.calculator import XTB
from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
xyz_path = '../xyz_structures/'
comp1 = os.path.join(xyz_path, 'holy_water.xyz' )
comp2 = os.path.join(xyz_path, 'Nd.xyz' )
comp3 = os.path.join(xyz_path, 'NO3-.xyz')
comp4 = os.path.join(xyz_path, 'OH-.xyz')
comp5 = os.path.join(xyz_path, 'Water.xyz')

""" Possible solvation box
System  Nd, NO3, OH, H2O, Total_atoms
100     1,  3,   0,  35,  118  ?
110     1,  2,   1,  36,  117
111     1,  1,   2,  37,  117  ?

200     2,  6,   0,  30,  116  ?
210     2,  5,   1,  31,  116
211     2,  3,   3,  33,  116  ?
212     2,  1,   5,  34,  116

300     3,  9,   0,  26,  117  ?
310     3,  8,   1,  27,  116
311     3,  7,   2,  28,  118  ?
312     3,  5,   4,  29,  118
313     3,  3,   6,  31,  117  ?
314     3,  1,   8,  32,  117

400     4, 12,   0,  22,  118  ?
410     4, 11,   1,  20,  118
411     4,  9,   3,  21,  117  ?
412     4,  6,   6,  23,  117  ?
413     4,  3,   9,  26,  117

500     5, 15,   0,  17,  116  ?
510     5, 12,   3,  18,  117  ?
511     5,  8,   7,  20,  117
512     5,  6,   9,  21,  117  ?
513     5,  3,  12,  23,  116

600     6, 18,   0,  13,  117  ?
610     6, 15,   3,  11,  118
611     6, 12,   6,  12,  117  ?
612     6,  9,   9,  14,  117
613     6,  6,  12,  18,  117  ?
614     6,  3,  15,  20,  117
"""

input_molecules = [ comp1 ] + [ comp2, comp3, comp4, comp5 ]
input_num_of_molecules = [1] + [ 1,  1,   2,  37,  ]
# Remove 0 num of molecule
correct_input = [ [inp_mol,inp_num] for inp_mol, inp_num in zip(input_molecules, input_num_of_molecules) if inp_num>0 ]
input_molecules = [ i[0] for i in correct_input ]
input_num_of_molecules = [ i[1] for i in correct_input ]
# Make the box
box0 = [ 6,6,6, 14,14,14 ]
input_constraint_type =  ['at_position'] + [ 'in_box' for i in range(len(input_molecules)-1) ]
input_constraint_value = [ () ] + [ box0 for i in range(len(input_molecules)-1) ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=(20,20,20),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

# Constraint
bond_pairs = cluster.compute_system_bond_pair()
bond_constraint = FixBondLengths( bond_pairs )
atom_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.index<700 ]) # water environment
#ase_constraint = [ atom_constraint, bond_constraint ]

print( "Step 2: Setting calculator" )
# Set the way to compute energy
model_path = 'mace-mh-1.model'
ase_calculator = mace_mp(model=model_path, dispersion='True', default_dtype="float32", device='cuda', head='omat_pbe')

# for ASE
#specific_charges = np.array([0.4,-0.8,0.4]*233 + [2.4] + [-0.4,-0.4,-0.4,0.4] + [0.1,-0.9]*2 + [0.4,-0.8,0.4]*37, dtype=np.float64 )
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg='from_structure_file',
                            coarse_calc_step=200, coarse_calc_fmax=2, coarse_calc_constraint=atom_constraint)
#geo_opt_parameter = dict(fmax=0.15, steps=500, ase_constraint=atom_constraint, Dual_stage_optimization=dict(fmax=0.05, steps=500) )
geo_opt_parameter = dict(fmax=0.05, steps=500 )

computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = geo_opt_parameter, 
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Full' ,#'Simple',
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=10, limit=40, max_iteration=5, initial_population_scaler=3,
                      ga_interval=2, ga_parents=5, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      apply_algorithm = 'ABC_GA',
                      #if_clip_candidate = True,  
                      early_stop_parameter = {'Max_candidate':5000},
                      )
optimization.run(print_interval=1)

