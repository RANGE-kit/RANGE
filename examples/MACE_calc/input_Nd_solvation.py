# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
import matplotlib.pyplot as plt
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
comp1 = os.path.join(xyz_path, 'Nd.xyz' )
comp2 = os.path.join(xyz_path, 'TODGA.xyz')
comp3 = os.path.join(xyz_path, 'NO3-.xyz')
comp4 = os.path.join(xyz_path, 'Water.xyz')

input_molecules = [comp1, comp2, comp3, comp4]
input_num_of_molecules = [1,1, 3,5]

box_region = [-5,-5,-5, 5,5,5]
input_constraint_type = ['at_position', 'in_box', 'in_box', 'in_box']
input_constraint_value = [(0,0,0,0,0,0), box_region, box_region, box_region ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        #pbc_box=(20.26028, 20.26028, 32.13928), # For BaTiO3 slab
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print( "Step 2: Setting calculator" )
# Set the way to compute energy

# for ASE
model_path = 'mace-mpa-0-medium.model' #'mace-mh-1.model'
ase_calculator = mace_mp(model=model_path, dispersion=False, default_dtype="float64", )
                         #device='cpu', head='omol')

# Constraint
bond_pairs = cluster.compute_system_bond_pair()
bond_constraint = FixBondLengths( bond_pairs )
atom_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.symbol == 'Nd'])
ase_constraint = [ atom_constraint, bond_constraint ]

coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=atom_constraint)
#geo_opt_parameter = dict(fmax=0.05, steps=100, ase_constraint=ase_constraint)
geo_opt_parameter = dict(fmax=0.5, steps=20, ase_constraint=ase_constraint, Dual_stage_optimization=dict(fmax=0.1, steps=20) )

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
                      colony_size=10, limit=40, max_iteration=1, initial_population_scaler=2,
                      ga_interval=1, ga_parents=5, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      apply_algorithm = 'ABC_GA',
                      if_clip_candidate = True,  
                      early_stop_parameter = {'Max_candidate':1000},
                      )
optimization.run(print_interval=1)

print( "Step 4: See results: use analysis script" )
