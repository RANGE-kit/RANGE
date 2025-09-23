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
#from xtb.ase.calculator import XTB
#from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input to assign the XYZ structure files
input_molecules = ['../xyz_structures/GraOxide.xyz', '../xyz_structures/HMIM.xyz' ]
input_num_of_molecules = [1, 6]
input_constraint_type = [ 'at_position', 'on_surface' ]
input_constraint_value = [ (), (0, (1.8,2.4), 30, 11) ]

print( "Step 1: Setting cluster" )
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        #pbc_box=(22.90076, 23.00272, 31.95000),  # Use this to consider PBC structures
                        )
cluster.init_molelcules()
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()  # Generate modeling setting

print( "Step 2: Setting calculator" )
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=20, coarse_calc_fmax=5, coarse_calc_constraint=None)
# Do not change the input part name "{input_script}" or name of output log "job.log". They are tags for the code.
exe_path = 'gmx'
calculator_command_line = exe_path + " -f {input_script} -c data-in.gro -p top.top -o run ; mdrun -s run.tpr > job.log "
geo_opt_control_line = dict(method='GROMACS', input='input-gmx.mdp') 

# Put all together for my calculation part
computation = energy_computation(templates = cluster_template,      # From previous definitions
                                 go_conversion_rule = cluster_conversion_rule,   # From previous definitions
                                 calculator = calculator_command_line,       
                                 calculator_type = 'external',          
                                 geo_opt_para = geo_opt_control_line,  
                                 if_coarse_calc = True,             # Do we want to pre-optimize using coarse optimizer before the fine optimizer
                                 coarse_calc_para = coarse_opt_parameter,   # The coarse optimizer setting
                                 #save_output_level = 'Full',        # How many output files to save? 'Full' means everything. 'Simple' means less output files are saved.
                                 )

output_folder_name = 'results'  # This folder will be created to keep output files.
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,  # From previous definitions
                      colony_size=10,            # The number of bee in ABC algorithm
                      limit=20,                 # The upper threshold to convert a bee to scout bee
                      max_iteration=100,         # The max number of iterations
                      ga_interval=1,            # The interval iteration to call GA algorithm
                      ga_parents=5,             # The number of bees to mutate (must be no more than colony_size)
                      mutate_rate=0.5, mutat_sigma=0.05,            # The mutation factor in GA
                      output_directory = output_folder_name,        # Save results to this folder
                      #restart_from_pool = 'structure_pool.db',      # Restart option: read previous results from either a database (XX.db) or a directory (results). None (default) means a fresh start.
                      )
optimization.run(print_interval=1)  # Start running. Print information with given frequency. The pool info is return for direct analysis

print( "Step 4: See results: use analysis script" )  # If pool info is not analyzed here. We can use the analysis script after finishing this calculation.
