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
from mace.calculators import mace_mp #MACECalculator, mace_anicc, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
xyz_path = '../xyz_structures/'
zeolite = os.path.join(xyz_path, 'Zeolite.xyz' )
dopant = os.path.join(xyz_path, 'Cu.xyz')

input_molecules = [zeolite, dopant]
input_num_of_molecules = [1,1]

# Find all atoms that can be replaced. In this case, all Si atoms
from ase.io import read
zeo = read(zeolite)
replace_atoms_indice = [ at.index for at in zeo if at.symbol=='Si'] 

input_constraint_type = ['at_position', 'replace']
input_constraint_value = [(), replace_atoms_indice ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=( 20.022, 19.899, 13.383 ),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# for ASE
model_path = '../mace-mpa-0-medium.model'
ase_calculator = mace_mp(model=model_path, dispersion=True, default_dtype="float64", device='cpu')

# Constraint
ase_constraint = None 
geo_opt_parameter = dict(fmax=0.1, steps=3, ase_constraint=ase_constraint)

computation = energy_computation(templates = cluster_template,
                                 go_conversion_rule = cluster_conversion_rule,
                                 calculator = ase_calculator,
                                 calculator_type = 'ase',
                                 geo_opt_para = geo_opt_parameter,
                                 if_coarse_calc = False,
                                 save_output_level = 'Simple',
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=10, limit=40, max_iteration=5, initial_population_scaler=2,
                      ga_interval=2, ga_parents=5, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      early_stop_parameter = {'Max_candidate':30},
                      )
optimization.run(print_interval=1)

print( "Step 4: See results: use analysis script" )

