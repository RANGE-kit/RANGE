from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import LJQ_calculator, RigidBody_filter

from ase.optimize import BFGS, FIRE
from ase.io import read, write
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLengths

import numpy as np
import matplotlib.pyplot as plt
import os


atoms = read( 'start.xyz' )

# -------------- Components ------------------
xyz_path = '../xyz_structures/'
comp1 = os.path.join(xyz_path, 'holy_water.xyz' )
comp2 = os.path.join(xyz_path, 'Nd.xyz' )
comp3 = os.path.join(xyz_path, 'NO3-.xyz')
comp4 = os.path.join(xyz_path, 'OH-.xyz')
comp5 = os.path.join(xyz_path, 'Water.xyz')

input_molecules = [ comp1 ] + [ comp2, comp3, comp4, comp5 ]
input_num_of_molecules = [1] + [ 1,  1,   2,  37,  ]
#input_num_of_molecules = [0] + [ 1,  0,   0,  10,  ]

# Remove 0 num of molecule
correct_input = [ [inp_mol,inp_num] for inp_mol, inp_num in zip(input_molecules, input_num_of_molecules) if inp_num>0 ]
input_molecules = [ i[0] for i in correct_input ]
input_num_of_molecules = [ i[1] for i in correct_input ]
# Make the box
box0 = [ 6,6,6, 14,14,14 ]
input_constraint_type =  ['at_position'] + [ 'in_box' for i in range(len(input_molecules)-1) ]
input_constraint_value = [ () ] + [ box0 for i in range(len(input_molecules)-1) ]

# Set the cluster structure
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=(20,20,20),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

# Set constraint
bond_pairs = cluster.compute_system_bond_pair()
bond_constraint = FixBondLengths( bond_pairs )
atom_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.index<700 ]) # water environment
ase_constraint = [ atom_constraint, bond_constraint ]
atoms.set_constraint( atom_constraint )

# Calculator with charge
specific_charges = np.array([0.4,-0.8,0.4]*233 + [2.4] + [-0.4,-0.4,-0.4,0.4] + [0.1,-0.9]*2 + [0.4,-0.8,0.4]*37, dtype=np.float64 )
coarse_calc = LJQ_calculator(cluster_template, charge=specific_charges, epsilon='UFF', sigma='UFF', cutoff=6)

atoms.calc = coarse_calc

rigid_atoms = RigidBody_filter(atoms, cluster_template)

#dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
dyn = FIRE(rigid_atoms, maxstep=0.5) #, logfile=dyn_log )
dyn.run( fmax=20, steps=100 )

write( 'structure_optimized.xyz', atoms )