# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_py.ga_abc import GA_ABC
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import energy_computation

import numpy as np
#import matplotlib.pyplot as plt
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
carbon = '../xyz_structures/C.xyz'

input_molecules = [carbon]
input_num_of_molecules = [12]
 
input_constraint_type = ['in_sphere_shell'] # in_box
input_constraint_value = [(0,0,0,11,11,11)] #[(0,0,0, 11,11,11, 4,4,4, 7,7,7) ]

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
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=None)

# for ASE
ase_calculator = XTB(method="GFN2-xTB", max_iterations=1000, electronic_temperature=2000 ) 
geo_opt_parameter = dict(fmax=0.2, steps=200)
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = geo_opt_parameter, # None = single point calc, 
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Full',
                                 )

"""
# for external xTB
xtb_exe_path = '/Users/d2j/Downloads/xtb-6.7.1/bin/xtb.exe'
calculator_command_line = xtb_exe_path + " --gfnff  {input_xyz} --etemp 2500 --opt normal --cycles 100 --iterations 1000 "
geo_opt_control_line = dict(method='xTB')
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = calculator_command_line,
                                 calculator_type = 'external', 
                                 geo_opt_para = geo_opt_control_line ,
                                 # Below are for coarse optimization
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Simple',
                                 )
"""
# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5, limit=10, max_iteration=5, initial_population_scaler=5,
                      ga_interval=2, ga_parents=5, mutate_rate=0.5, mutat_sigma=0.03,
                      output_directory = output_folder_name,
                      # Restart option
                      restart_from_pool = 'structure_pool.db',
                      apply_algorithm = 'ABC_GA',
                      )
all_x, all_y, all_name = optimization.run(print_interval=1, if_return_results=True )

print( "Step 4: See results: use analysis script" )
