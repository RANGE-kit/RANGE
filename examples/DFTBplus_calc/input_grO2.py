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
from ase.constraints import FixAtoms
from ase import Atoms

from xtb.ase.calculator import XTB
from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
xyz_path = '../../structures/'
gr = os.path.join(xyz_path, 'GrH-44.xyz' )
o2 = os.path.join(xyz_path, 'O2.xyz')

input_molecules = [ gr, o2 ]
input_num_of_molecules = [1,1]

#input_constraint_type = ['at_position','on_surface']
#input_constraint_value = [(),(0, (1.5,2.0), 0, 1) ]
input_constraint_type =  [ 'at_position', 'in_box' ]
input_constraint_value = [ (), [-3.53, -2.44, 0.8, 2.83, 1.22, 1.5] ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        #pbc_box=(22.90076, 23.00272, 31.95000),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Constraint
ase_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.symbol != 'O'])

coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=10, coarse_calc_fmax=10, coarse_calc_constraint=ase_constraint)

"""
# for ASE
#ase_calculator = XTB(method="GFN2-xTB") 
model_path = '/global/homes/z/zdf/software/mace-mpa-0-medium.model'
ase_calculator = mace_mp(model=model_path, dispersion=False, default_dtype="float64", device='cuda')

geo_opt_parameter = dict(fmax=0.02, steps=500)
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

"""
# for external 
exe_path = '/global/homes/z/zdf/bin/dftb+'
calculator_command_line = exe_path + "   > job.log " 
geo_opt_control_line = dict(method='DFTB+', input='input_dftbp_template.hsd')
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

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=20, limit=40, max_iteration=10, 
                      ga_interval=1, ga_parents=10, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      )
optimization.run(print_interval=1, if_return_results=False)

print( "Step 4: See results: use analysis script" )
