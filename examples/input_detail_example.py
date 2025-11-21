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
from ase.io import read, write

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
from xtb.ase.calculator import XTB
#from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

""" ----------------------------------------------------
Provide user input to assign the XYZ structure files, for example:
""" 
water = 'xyz_structures/Water.xyz'
methane = 'xyz_structures/methane.xyz'
organic_sub = 'xyz_structures/Substrate.xyz'
single_atom = 'xyz_structures/Single_atom.xyz'
slab_surface = 'xyz_structures/Slab_BaTiO3_7layer.xyz'
copper_13 = 'xyz_structures/Cu13.xyz'
co2 = 'xyz_structures/CO2.xyz'
# Or use ASE atoms

""" ----------------------------------------------------
Provide the number of molecules considered for every type of molecule
In this case, we consider 2 CO2 molecules and 5 water molecules
"""
input_molecules = [co2, water]
input_num_of_molecules = [2, 5]

""" ----------------------------------------------------
Provide the constraint types by:

at_position 
    input parameters: X,Y,Z, (Euler_X, Euler_Y, Euler_Z) 
    All parameters are float type number. 
    To add the molecule to a certain position. X,Y,Z are the geometric center of the molecule. Must be assigned by user.
    Euler_X, Euler_Y, Euler_Z are the orientation of the molecule. If not assigned by user, molecule's orientation will be a varaible to be optimized.
    Example: (0,0,0, 90, 90, 0)
    
in_box
    input parameter: xlo,ylo,zlo,xhi,yhi,zhi, (xlo,ylo,zlo,xhi,yhi,zhi)
    All parameters are float type number. 
    To add the molecule inside a box region defined by lower limit (xlo,ylo,zlo) and upper limit (xlo,ylo,zlo)
    If the second box is defined, the molecule is added so that it is outside the second box.
    Example: (0,0,0, 10,10,10, 3,3,3, 7,7,7)
        
in_sphere_shell
    Input parameter: X,Y,Z, R_x, R_y, R_z, (dR_ratio)
    All parameters are float type number. dR_ratio must be between (0,1).
    To add the molecule inside a spherical region defined by the center of sphere (X,Y,Z) and three primary axis length (Rx, Ry, Rz).
    If Rx, Ry, Rz is not the same, this will be an ellipsoid region.
    If another parameter dR_ratio is provided, the molecule is added to the spherical shell with outer R as Rx and inner R as Rx*(1-dR_ratio) for x. Same for y and z.
    Example: ( 0,0,0, 3,4,5, 0.1 )

on_surface
    input parameter: ID_substrate, (lo,hi), ID_atom_on_surf, ID_atom_orientation
    ID_substrate, ID_atom_on_surf, ID_atom_orientation are integer number. lo and hi are float.
    To add molecule on the surface of a substrate.  
    ID_substrate is the index of the molecule in the template. Currently must set the substrate molecule to the first one, i.e. with ID=0
    (lo, hi) define the limit of adsorption distance between the adsorbate and adsorbent.
    ID_atom_on_surf and ID_atom_orientation define the atom index on the surface and the atom index to define the orientation of adsorbate.
    Example: (0, (1.9, 2.1), 0, 6)
    
in_pore
    input parameter: ID_substrate, ID_atom_considered, grid spacing 
    ID_substrate in integer, ID_atom_considered is a list of integer, grid spacing is float
    To add molecules in the space defined by the grid points from ID_atom.
    Example: ( 0, (1,2,5,8,9), 0.1 )
    
micelle or layer
    input parameter: (X Y Z),(X Y Z),(X Y Z),(X,Y,Z),spacing,n1,n2 for layer, or A,B,C,(X Y Z),spacing,n1,n2 for ellipsoid
    X,Y,Z,A,B,C,spacing are float. n1, n2 are int
    To add molecules on a planar surface or ellipsoid surface using an uniform grid 
    Example: ( (0,0,0), (1,0,0), (0,1,0), (0,0,1), 1.5, 2,4 ) or (3,4,5, (0,0,0), 1.8, 3,5)

replace:
    input parameter: tuple/list of integers 
    To replace certains molecules in the substrate by this atom/molecule. The substrate is the first molecule. 
    list of integers indicates the index of available atoms in the substrate to be replaced.
    Example: [1,4,7,9,10,16,19]
"""
input_constraint_type = [ 'in_sphere_shell', 'in_sphere_shell' ]
input_constraint_value = [ (0,0,0,4,4,4), (0,0,0,5,5,5,0.2) ]

""" ----------------------------------------------------
Set the cluster structure and initialize the boundary conditions. All input is defined above.
"""
print( "Step 1: Setting cluster" )
cluster = cluster_model(input_molecules, input_num_of_molecules, 
                        input_constraint_type, input_constraint_value,
                        # pbc_box=(22.90076, 23.00272, 31.95000),  # Use this to consider PBC structures
                        # pbc_applied=(True,True,True) # Use this to consider no-box (pbc_box is not assigned) or Dirichlet-BC (pbc_box is assigned but False appears here)
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()  # Generate modeling setting

""" ----------------------------------------------------
Set the way to compute energy once we have the molecular structures.

The coarse optimizer uses Lennard Jones force field plus Columb interaction, and it fixes internal degree of freedom (i.e. rigid body)
By default we use UFF parameter and no atomic charge. No constraint during optimization (e.g. no frozen atom)

Then we need to pick a more accurate way to compute energy: ASE calculator or external code

"""
print( "Step 2: Setting calculator" )
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=10, coarse_calc_fmax=10, coarse_calc_constraint=None )

# We can use ASE calculator python interface
ase_calculator = XTB(method="GFN2-xTB") #TBLite(method="GFN2-xTB", verbosity=-1) # Use semi-empirical
#ase_calculator = mace_mp(model='small', dispersion=False, default_dtype="float64", device='cuda') # Or use MACE force field
geo_opt_parameter = dict(fmax=0.2, steps=20) # This is the ASE keywords for BFGS optimizer.

# Put all together for my calculation part
computation = energy_computation(templates = cluster_template,      # # From previous definitions
                                 go_conversion_rule = cluster_conversion_rule,   # # From previous definitions
                                 calculator = ase_calculator,       # The fine optimizer
                                 calculator_type = 'ase',           # What calculator will we use. 'ase' means use ASE calculator. 
                                 geo_opt_para = geo_opt_parameter,  # if set to "None", we will do only single point calc with the fine optimizer                                 
                                 if_coarse_calc = True,             # Do we want to pre-optimize using coarse optimizer before the fine optimizer
                                 coarse_calc_para = coarse_opt_parameter,   # The coarse optimizer setting
                                 save_output_level = 'Full',        # How many output files to save? 'Full' means everything. 'Simple' means less output files are saved.
                                 check_structure_sanity = None,     # If check structure for unreasonable bonds after optimization. None = only check atom distance. = cluster.internal_connectivity
                                 )
"""
# We can also use an external software. For example:
# for external xTB:
    xtb_exe_path = '/Users/d2j/Downloads/xtb-6.7.1/bin/xtb.exe'
    calculator_command_line = xtb_exe_path + " --gfn2  {input_xyz} --opt normal --cycles 100 --iterations 1000 "
    geo_opt_control_line = dict(method='xTB')  
# for external CP2K:
    calculator_command_line = " srun shifter --entrypoint cp2k -i  {input_script}  -o job.log "
    geo_opt_control_line = dict(method='CP2K', input='input_CP2K')
# for external Gaussian:
    calculator_command_line = " g16 < {input_script}  > job.log "
    geo_opt_control_line = dict(method='Gaussian', input='input_gaussian_template') # This input will be your gaussian input script
# for anything else in general: (This can be flexible to add other software as needed.)
    calculator_command_line = " XXXX {input_script} XXXX"  
    geo_opt_control_line = dict(method='User', get_energy='XXX', get_structure='XXX')  
    # Three requirements: (1) "{input_script}" must exist, which corresponds to the XYZ input file (2) "get_energy" returns energy (3) "get_structure" returns structure file name
# Then in the above energy_computation setting, use:
    calculator = calculator_command_line ,
    calculator_type = 'external',
    geo_opt_para = geo_opt_control_line ,
We currently provided external examples for LAMMPS, GROMCAS, xTB, DFTB+, CP2K, ORCA, Gaussian, SPARC.
See example directory for details
"""

""" ----------------------------------------------------
Now we know what structures we are making and how to compute their energies. 
We can put together and run the algorithm to find good structures
"""
output_folder_name = 'results'  # This folder will be created to keep output files.
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,  # From previous definitions
                      colony_size=20,            # The number of bee in ABC algorithm
                      limit=40,                 # The upper threshold to convert a bee to scout bee
                      max_iteration=50,         # The max number of iterations
                      initial_population_scaler=5,# How many initial guess to be made before search?
                      ga_interval=2,            # The interval iteration to call GA algorithm
                      ga_parents=10,             # The number of bees to mutate (must be no more than colony_size)
                      mutate_rate=0.5, mutat_sigma=0.05,            # The mutation factor in GA
                      output_directory = output_folder_name,        # Save results to this folder
                      output_header = 'compute_',                   # Default computing ID header. 
                      output_database = 'structure_pool.db',        # Default database name. 
                      restart_from_pool = 'structure_pool.db',      # Restart option: read previous results from either a database (XX.db) or a directory (results). None (default) means a fresh start.
                      restart_strategy = 'lowest',                  # Restarting method: default is "lowest" (pick the lowest candidates). Can also be "random".
                      apply_algorithm = 'ABC_GA',                   # "ABC_GA" algorithm to use (default). Can also be 'ABC_random' or 'GA_native' for backward compatibility.
                      if_clip_candidate = True,                     # Clip the generated candidates (default).
                      early_stop_parameter = None,                  # Early stop using key: 'Max_candidate', 'Max_ratio', or 'Max_lifetime'. Default is None (no early stop)
                      )
optimization.run(print_interval=1, if_return_results=False)  # Start running. Print information with given frequency. Ignore log and return output history

print( "Use example/analysis_output_energy.py to analyze/summarize results, or create your own" )  # If pool info is not analyzed here. We can use the analysis script after finishing this calculation.
