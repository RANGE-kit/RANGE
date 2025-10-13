# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""

from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
import os

from sparc import SPARC


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
nh3 = '../xyz_structures/NH3.xyz'
substrate = '../xyz_structures/BaTiO3-cub-7layer-TiO.xyz' 

input_molecules = [substrate, nh3]
input_num_of_molecules = [1, 3]

input_constraint_type = ['at_position','in_box']
input_constraint_value = [(0,0,20,0,0,0),(-2.5,-2.5,27.5, 2.5,2.5,32.5) ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        pbc_box=(20.26028, 20.26028, 50), 
                        pbc_applied = (True, True, False),
                        )
cluster.init_molelcules()
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Set the way to compute energy
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=None)

# for ASE
ase_calculator = SPARC(h=0.18, kpts=(1, 1, 1), xc="pbe", directory="log-ase")
geo_opt_parameter = dict(fmax=0.1, steps=200)
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = geo_opt_parameter, # None = single point calc, 
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Simple',
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=10, max_iteration=5, initial_population_scaler=2,
                      ga_interval=2, ga_parents=3, mutate_rate=0.5, mutat_sigma=0.03,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      apply_algorithm = 'ABC_GA',
                      )
all_x, all_y, all_name = optimization.run(print_interval=1, if_return_results=True )

print( "Step 4: See results: use analysis script" )
