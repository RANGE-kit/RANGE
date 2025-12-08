# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_go.cluster_model import cluster_model
from RANGE_go.utility import structure_difference

import numpy as np
#import matplotlib.pyplot as plt
import os, argparse

from ase.io import read, write
from ase.db import connect
from ase import neighborlist as ngbls

from tqdm import tqdm


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
-----------------
| Input setting  |
-----------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--db',    type=str, default='structure_pool.db', help='db file to read')
parser.add_argument('--input', type=str, default='input_RANGE.py', help='RANGE input')
parser.add_argument('--group', type=int, nargs="+", default='-1', help='apply grouping analysis. Default: not perform any grouping')
parser.add_argument('--align', type=int, nargs="+", default='0', help='Atom ID to align structure for gas phase modeling. Default: no align')
parser.add_argument('--shift', type=float, nargs="+", default='0', help='Translate dX dY dZ of structure for surface modeling. Default: no translation')
args = parser.parse_args()
#print( args )

# Here is case-dependent conditions to check structure. Modified based on RANGE_go.utility.check_structure.
# Each list indicates a molecule in the input molecules (input_molecules). The elements in the list will be used for connectivity check. Empty list means no check.
check_atom_symbols = [ ['C','N'],['C','O','H'],['H','O'] ]  # Use empty list for no-check

# Additional constraints for checking a good/bad structure can be set on line 177: Other customized conditions? By default, no other condition is considered.

"""
------------------------------------------------------------
| Cluster information should agree with generation setting  |
------------------------------------------------------------
Can copy/paste from the corresponding RANGE search input
"""
xyz_path = '../'
comp1 = os.path.join(xyz_path, 'mace-opted-box32singlelayer-C3N4-surf-Pt4.xyz' )
comp2 = os.path.join(xyz_path, 'C3H6.xyz')
comp3 = os.path.join(xyz_path, 'H2.xyz')
input_molecules = [ comp1, comp2, comp3 ]
input_num_of_molecules = [1,1,1]

# We just need the cluster class to use its functions/vars for the cluster components, so here we just make a fake cluster
input_constraint_type = [ 'at_position' for n in input_molecules ]
input_constraint_value = [ () for n in input_molecules ]
cluster = cluster_model(input_molecules, input_num_of_molecules, input_constraint_type, input_constraint_value,
                        pbc_box=(21.404922 , 24.706836 , 30.27529),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()


"""
Helper functions
"""
def alignment(mol, args_input):
    if args_input==0: # No change
        pass
    elif len(args_input) in [3,4] :  
    # 3 = two atom ID for X direction, then one for plane definition and Y direction
    # 4 = two atom ID for X direction, then one for plane definition, and one for Z direction
        x1_atom_id = args_input[0]
        x2_atom_id = args_input[1]
        x3_atom_id = args_input[2]

        # Rotate X1-->X2 into positive X-axis
        mol.rotate( mol.positions[x2_atom_id] - mol.positions[x1_atom_id] , (1,0,0), center=(0,0,0) )
        # Move center atom to (0,0,0)
        center = ( mol.positions[x1_atom_id] + mol.positions[x2_atom_id] )*0.5
        mol.translate( -center ) 

        # X1, X2, and X3 plane
        pos = mol.get_positions()
        u = pos[x1_atom_id] - pos[x3_atom_id]
        v = pos[x2_atom_id] - pos[x3_atom_id]
        surf_norm = np.cross(u, v)
        surf_norm = surf_norm/np.linalg.norm(surf_norm)

        if len(args_input) == 4:
            z1_atom_id = args_input[3]
            w = pos[z1_atom_id] - pos[x3_atom_id]
            if np.dot( surf_norm, w ) <0:
                surf_norm = -surf_norm

        # rotate along X to make surf_norm -> +Z
        _, ny, nz = surf_norm
        theta = np.degrees( np.arctan2(ny, nz) )     # angle needed
        mol.rotate( theta, 'x' )  # rotate along X

        if len(args_input) == 3:
            if mol.positions[x3_atom_id][2]<0:
                mol.rotate( 180, 'x' )  # rotate along X
    else:
        raise ValueError('Keyword align has a wrong input')
    return mol

def read_RANGE_input(inp):
    with open( inp, 'r' ) as f1:
        lines = f1.readlines()
    lines1 = [ l for l in lines if "input_molecules" in l ]
    lines2 = [ l for l in lines if "input_num_of_molecules" in l ]
    print( lines1, lines2 )
    exit()


"""
-----------------------------------------------------------------
| Assign which atoms should be considered in connectivity check  |
-----------------------------------------------------------------
"""
# check_atom_symbols is setup at the begining of the script.

out = []
for c,n in zip(check_atom_symbols,input_num_of_molecules):
     out += [c]*n
check_atom_symbols = out

db_path = args.db 
sorted_clean_traj = [] #'structure_pool_sorted_clean.xyz'
ener, name = [],[]

print( "Start analysis..." )
db = connect(db_path)
#for nr, row in enumerate(db.select()):
for nr, row in tqdm(enumerate(db.select()), total=len(db), desc="Processing items"):
    atoms = row.toatoms() 
    e = row.data.output_energy

    if e<1e3:
        check_pass = True
    else:
        check_pass = False

    if check_pass:
        index_head, index_tail = -1,-1
        for n, molecule in enumerate(cluster.templates):
            index_head = index_tail +1  # point to the first atom in mol
            index_tail = index_head +len(molecule) -1 # point to the last atom in mol
            new_mol = atoms[ index_head: index_tail+1 ]
        
            if len(check_atom_symbols[n])>0 and len(new_mol)>1:
                # check these atoms
                check_atoms_index = [ at.index for at in new_mol if at.symbol in check_atom_symbols[n] ]

                connect_ref = cluster.internal_connectivity[n] # C in the input structure
                connect_ref = connect_ref[ np.ix_(check_atoms_index, check_atoms_index) ] # these rows and cols

                cutoffs = [ n for n in ngbls.natural_cutoffs(new_mol, mult=1.01) ]
                ngb_list = ngbls.NeighborList(cutoffs, self_interaction=False, bothways=True)
                ngb_list.update(new_mol)
                connect = ngb_list.get_connectivity_matrix(sparse=False) # C in the generated structure
                connect     = connect[ np.ix_(check_atoms_index, check_atoms_index) ] # these rows and cols
            
                #if np.array_equal(connect, connect_ref):
                if np.sum(np.abs(connect-connect_ref))!=0:
                    check_pass = False
                    break
                
    #  Perform geometry re-orientation before analysis, if needed
    if check_pass:
        atoms = alignment(atoms, args.align) 
    
        """
        ---------------------------------
        | Other customized conditions?  |
        ---------------------------------
        """
        pos = atoms.get_positions()
        #n = [ at.index for at in atoms if at.symbol=='N' ]
        #h = [ at.index for at in atoms if at.symbol=='H' ]
        #pos_n = np.mean(pos[n], axis=0)[2]
        #pos_h = np.mean(pos[h], axis=0)[2]
        #pos_cu_max = np.amax( pos[cu][:,2] )
        #pos_cu_min = np.amin( pos[cu][:,2] )
        #if pos_n > pos_h :
        #    check_pass = False
        """
        --------------------------------
        """

    if check_pass:
        # Final output
        m = row.data.compute_name
        sorted_clean_traj.append( atoms )
        ener.append( row.data.output_energy )
        name.append( row.data.compute_name )

# sort energy and write
print( "Sort energy" )
sorted_idx = np.argsort(ener)
tags = None

"""
--------------------------------------------------------------------------
| Perform more detailed ranking based on energy and geometry, if needed   |
--------------------------------------------------------------------------
"""
# if needed, we can do the similarity analysis to clean more
if isinstance(args.group, int):
    args.group = [ args.group ]
if args.group[0]>=0: 
    use_components_index = args.group
    use_atoms_index = [] # use these atoms to comparison
    idx_point = 0
    for im, mol in enumerate(cluster_template):
        idx = np.arange(len(mol)) + idx_point
        if cluster.global_molecule_index[im] in use_components_index: 
            use_atoms_index += idx.tolist()
        idx_point += len(mol)

    atoms_ref = sorted_clean_traj[ sorted_idx[0] ]
    atoms_ref = atoms_ref[use_atoms_index] # To reduce comp cost
    dmat_ref = atoms_ref.get_all_distances(mic=True)
    ener_ref = ener[ sorted_idx[0] ]
    dmat_tag, ener_tag = [], []
    for n, atoms in enumerate(sorted_clean_traj):
        atoms = atoms[use_atoms_index]
        dmat = atoms.get_all_distances(mic=True)
        dmat_diff = dmat - dmat_ref
        #dmat_diff = dmat_diff[np.ix_(use_atoms_index, use_atoms_index)] # these rows and cols
        i, j = np.triu_indices(dmat_diff.shape[0], k=1)
        d = dmat_diff[i,j]
        #dmat_tag.append( np.linalg.norm(d)*np.sign(d.mean()) )
        dmat_tag.append( np.linalg.norm(d) )
        # energy
        ener_diff = ener[n] - ener_ref
        ener_tag.append( ener_diff )
    ener_tag = np.round( ener_tag , 2 ) # if diff is within 0.01, they are the same

    tags =  np.transpose([ ener_tag, dmat_tag ])
    sorted_idx = np.lexsort([tags[:, i] for i in reversed(range(tags.shape[1]))]) # rank 1st item, then 2nd, etc..

"""
---------------
| FInal write |
---------------
"""
print( "Write output" )
output_traj, output_ener, output_name = [],[],[]
write_seprate_lines = 1e9
with open('energy_summary_sorted_clean.log','w') as f1:
    for ii, i in enumerate(sorted_idx):
        if tags is None:
            line = f"{ii:8d} {ener[i]:16.10g}   {name[i]}\n"
        else:
            line = f"{ii:8d} {ener[i]:16.10g} {tags[i][0]:8.4g} {tags[i][1]:16.10g}   {name[i]}\n"
            if tags[i][0] != write_seprate_lines:
                line = "="*20 + '\n' + line
            write_seprate_lines = tags[i][0]
        f1.write(line)
        atoms = sorted_clean_traj[i]
        atoms.translate( args.shift )

        atoms.wrap()
        output_traj.append( atoms )
        #output_ener.append( ener[i] )
        #output_name.append( name[i] )

# Write cleaned traj
write( 'structure_pool_sorted_clean.xyz', output_traj )

