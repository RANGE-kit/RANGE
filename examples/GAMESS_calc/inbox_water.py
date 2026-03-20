# -*- coding: utf-8 -*-
"""
GAMESS calculator example for RANGE
"""

from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
import os

print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input to assign the XYZ structure files
water = '../xyz_structures/Water.xyz'
input_molecules = [water]
input_num_of_molecules = [3]
input_constraint_type = [ 'in_box' ]
input_constraint_value = [ (0,0,0, 5,5,5) ]

print( "Step 1: Setting cluster" )
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()  # Generate modeling setting

print( "Step 2: Setting calculator" )
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                            coarse_calc_step=10, coarse_calc_fmax=10, coarse_calc_constraint=None)
# Do not change the input part name "{input_script}" or name of output log "job.log". They are tags for the code.
# Adjust the rungms path and version/nprocs as needed for your GAMESS installation
gamess_path = '/Users/federicozahariev/GitHub/GAMESS/gamess_Jan_2026'
calculator_command_line = f" {gamess_path}/rungms {{input_script}} 01 1 > job.log "
geo_opt_control_line = dict(method='GAMESS', input='input_gamess_template')

# Put all together for my calculation part
computation = energy_computation(templates = cluster_template,
                                 go_conversion_rule = cluster_conversion_rule,
                                 calculator = calculator_command_line,
                                 calculator_type = 'external',
                                 geo_opt_para = geo_opt_control_line,
                                 if_coarse_calc = True,
                                 coarse_calc_para = coarse_opt_parameter,
                                 )

output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=5,
                      limit=20,
                      max_iteration=10,
                      ga_interval=2,
                      ga_parents=3,
                      mutate_rate=0.5, mutat_sigma=0.03,
                      output_directory = output_folder_name,
                      )
optimization.run(print_interval=1)

print( "Step 4: See results: use analysis script" )
