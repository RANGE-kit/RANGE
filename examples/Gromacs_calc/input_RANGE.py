from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import numpy as np
#import matplotlib.pyplot as plt
import os

#from ase.visualize import view
#from ase.visualize.plot import plot_atoms
#from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths
from ase import Atoms

#from ase.calculators.emt import EMT
#from tblite.ase import TBLite
#from xtb.ase.calculator import XTB
#from mace.calculators import mace_mp  
### MACECalculator, mace_anicc, mace_mp, mace_off


print("Step 0: Preparation and user input")
# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
xyz_path = '../xyz_structures/'
comp1 = os.path.join(xyz_path, 'box_holy_water.gro' )
comp2 = os.path.join(xyz_path, 'nd3.gro' )
comp3 = os.path.join(xyz_path, 'no3.gro')
comp4 = os.path.join(xyz_path, 'oh-.gro')
comp5 = os.path.join(xyz_path, 'water.gro')

""" Possible solvation box
System,  Nd,  NO3,  OH,  H2O,  Total_atoms
100,      1,    3,   0,   31,          106  #1
110,      1,    2,   1,   31,          104
111,      1,    1,   2,   32,          105  #2
200,      2,    6,   0,   26,          104  #3
210,      2,    5,   1,   27,          105
211,      2,    3,   3,   28,          104  #4
212,      2,    1,   5,   30,          106
300,      3,    9,   0,   22,          105  #5
310,      3,    8,   1,   23,          106
311,      3,    7,   2,   23,          104  #6
312,      3,    5,   4,   25,          106
313,      3,    3,   6,   26,          105  #7
314,      3,    1,   8,   27,          104
400,      4,   12,   0,   18,          106  #8
410,      4,   11,   1,   18,          104
411,      4,    9,   3,   20,          106  #9
412,      4,    6,   6,   22,          106  #10
413,      4,    3,   9,   24,          106
500,      5,   15,   0,   13,          104  #11
510,      5,   12,   3,   15,          104  #12
511,      5,    8,   7,   18,          105
512,      5,    6,   9,   19,          104  #13
513,      5,    3,  12,   21,          104
600,      6,   18,   0,    9,          105  #14
610,      6,   15,   3,   11,          105
611,      6,   12,   6,   13,          105  #15
612,      6,    9,   9,   15,          105
613,      6,    6,  12,   17,          105  #16
614,      6,    3,  15,   19,          105
"""

input_molecules = [ comp1 ] + [ comp2, comp3, comp4, comp5 ]
input_num_of_molecules = [1] + [ 6, 6, 12, 17, ]
# Remove 0 num of molecule
correct_input = [ [inp_mol,inp_num] for inp_mol, inp_num in zip(input_molecules, input_num_of_molecules) if inp_num>0 ]
input_molecules = [ i[0] for i in correct_input ]
input_num_of_molecules = [ i[1] for i in correct_input ]
# Make the box
box0 = [ 6,6,6, 14,14,14 ]
input_constraint_type =  ['at_position'] + [ 'in_box' for i in range(len(input_molecules)-1) ]
input_constraint_value = [ () ] + [ box0 for i in range(len(input_molecules)-1) ]

print( "Step 1: Setting cluster" )
# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=(20,20,20),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print('Num of atoms: ', len(cluster.system_atoms) )
mass_g = cluster.system_atoms.get_masses().sum()* 1.66053906660e-24  # mass in g
volumes =  20**3 * 1e-24  # Vol in cm3
print( 'Density in g/cm3 ', mass_g / np.asarray(volumes) )
    
# Constraint
#bond_pairs = cluster.compute_system_bond_pair()
#bond_constraint = FixBondLengths( bond_pairs )
#ase_constraint = [ atom_constraint, bond_constraint ]
# for ASE coarse opt
atom_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.index<932 ]) # water environment
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, #'from_structure_file',
                            coarse_calc_step=20, coarse_calc_fmax=1, coarse_calc_constraint=atom_constraint)

# for GROMACS opt
#calculator_command_line = " shifter --image docker:nersc/gromacs:23.2 gmx_mpi grompp -f {input_script}  -c data-in.gro  -p {input_topo} -o em.tpr -maxwarn 1 &&\
#        shifter --image docker:nersc/gromacs:23.2  gmx_mpi mdrun -deffnm em -ntomp 32 "
calculator_command_line = " shifter  gmx_mpi grompp -f {input_script}  -c data-in.gro  -p {input_topo} -o em.tpr -maxwarn 1 &&\
        shifter  gmx_mpi mdrun -deffnm em -ntomp 32 "
geo_opt_control_line = dict(method='GROMACS', mdp_file='/global/cfs/cdirs/m3269/difan/heavy-elements/tasks_no-ligand/min.mdp',topo_file='topo.top',)

computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator_type = 'external', 
                                 calculator = calculator_command_line,
                                 geo_opt_para = geo_opt_control_line, 
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 #save_output_level = 'Simple',
                                 check_structure_sanity = False,  # It is important to set False for dummy atom (e.g. tip4p)
                                 )

# Put together and run the algorithm
output_folder_name = 'results'
print( f"Step 3: Run. Output folder: {output_folder_name}" )
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=40, limit=40, max_iteration=100, initial_population_scaler=5,
                      ga_interval=2, ga_parents=20, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory = output_folder_name,
                      # Restart option
                      #restart_from_pool = 'structure_pool.db',
                      #apply_algorithm = 'ABC_GA',
                      #if_clip_candidate = True,  
                      early_stop_parameter = {'Max_candidate':10000},
                      )
optimization.run(print_interval=1)

