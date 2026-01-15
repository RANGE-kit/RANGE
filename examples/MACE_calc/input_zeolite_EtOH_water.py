from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
import matplotlib.pyplot as plt
import os

#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#from ase.io import read, write
from ase.constraints import FixAtoms
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
substrate = os.path.join(xyz_path, 'Zeolite.xyz' )
adsorb1 = os.path.join(xyz_path, 'Ethanol.xyz')
adsorb2 = os.path.join(xyz_path, 'Water_H+.xyz')
adsorb3 = os.path.join(xyz_path, 'Water.xyz')

input_molecules = [substrate,adsorb1,adsorb2,adsorb3]
input_num_of_molecules = [1,1,1,2]

pore_bound = (287,95,96,97,113,106,250,265,270,136,262,258,261)
input_constraint_type = ['at_position', 'in_pore', 'in_pore', 'in_pore']
input_constraint_value = [(), (0,pore_bound,0.2), (0,pore_bound,0.2), (0,pore_bound,0.2) ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=( 20.022, 19.899, 13.383 ),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Set the way to compute energy

# for ASE
model_path = '/ccsopen/home/d2j/software/downloaded_models/mace-mpa-0-medium.model'
ase_calculator = mace_mp(model=model_path, dispersion=True, default_dtype="float64", device='cpu')

# Constraint
ase_constraint = None 

geo_opt_parameter = dict(fmax=0.05, steps=200, ase_constraint=ase_constraint)
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=ase_constraint)

computation = energy_computation(templates = cluster_template,
                                 go_conversion_rule = cluster_conversion_rule,
                                 calculator = ase_calculator,
                                 calculator_type = 'ase',
                                 geo_opt_para = geo_opt_parameter,
                                 if_coarse_calc = True,
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Simple',
                                 if_check_structure_sanity = True,
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=20, limit=40, max_iteration=5000,
                      ga_interval=1, ga_parents=10, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      apply_algorithm = 'ABC_GA',
                      if_clip_candidate = True,
                      early_stop_parameter = {'Max_candidate':10000},
                      )
optimization.run(print_interval=1)

print( "Step 4: See results: use analysis script" )

