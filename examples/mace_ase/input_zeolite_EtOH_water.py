from RANGE_py.ga_abc import GA_ABC
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import energy_computation

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
xyz_path = '/global/cfs/cdirs/m4544/difan/global_optimization_RANGE/'
substrate = os.path.join(xyz_path, 'zeolite.xyz' )
adsorb1 = os.path.join(xyz_path, 'ethanol.xyz')
adsorb2 = os.path.join(xyz_path, 'water_H+.xyz')
adsorb3 = os.path.join(xyz_path, 'water.xyz')

input_molecules = [substrate,adsorb1,adsorb2,adsorb3]
input_num_of_molecules = [1,1,1,2]

input_constraint_type = ['at_position', 'in_box', 'in_box', 'in_box']
input_constraint_value = [(), (13.5, 2, 5,  16.5, 8, 8), (13.5, 2, 5,  16.5, 8, 8), (13.5, 2, 5,  16.5, 8, 8) ]


print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=( 20.022, 19.899, 13.383 ), 
                        )
cluster.init_molelcules()
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Set the way to compute energy

# for ASE
#ase_calculator = XTB(method="GFNFF")
#model_path = '/ccsopen/home/d2j/software/downloaded_models/2023-12-10-mace-128-L0_energy_epoch-249.model'
model_path = '/global/homes/z/zdf/software/mace-mpa-0-medium.model'
ase_calculator = mace_mp(model=model_path, dispersion=False, default_dtype="float64", device='cuda')

# Constraint
cluster_atoms = Atoms()
#index_Al = [at.index for at in cluster_atoms if at.symbol =='Al'][0]
ase_constraint = None #FixAtoms(indices=[at.index for at in cluster_atoms if at.symbol != 'Cu'])

geo_opt_parameter = dict(fmax=0.2, steps=200, ase_constraint=ase_constraint)
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                            coarse_calc_step=50, coarse_calc_fmax=10, coarse_calc_constraint=ase_constraint)

computation = energy_computation(templates = cluster_template,
                                 go_conversion_rule = cluster_conversion_rule,
                                 calculator = ase_calculator,
                                 calculator_type = 'ase',
                                 geo_opt_para = geo_opt_parameter,
                                 if_coarse_calc = True,
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Simple',
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=20, limit=40, max_iteration=50000,
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

