# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""


from RANGE_go.cluster_model import cluster_model
from RANGE_ts.utility import read_trajectory, structure_difference, alignment

import numpy as np
#import matplotlib.pyplot as plt
import os, argparse
from tqdm import tqdm

from ase.io import read, write
from ase import neighborlist as ngbls


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
------------------------------------------------------------
| Specific clean setting  |
------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='structure_pool.db', help='Trajectory input (db or xyz)')
parser.add_argument('--output', type=str, default='Frames_sorted_clean.xyz', help='Trajectory output name')
parser.add_argument('--log', type=str, default='Energy_summary_sorted_clean.log', help='Energy log output name')
parser.add_argument('--group', type=int, nargs="+", default='-1', help='apply grouping analysis. Input one or more integer indice of molecule ID.')
parser.add_argument('--align', type=int, nargs="+", default='0', help='Atom ID to align structure for gas phase modeling')
parser.add_argument('--shift', type=float, nargs="+", default='0', help='Translate dX dY dZ of structure for surface modeling')
args = parser.parse_args()
print( args )

# Here is case-dependent conditions to check structure. Modified based on RANGE_go.utility.check_structure
check_atom_connectivity = [ ['C','N'], ['C','O','H'], ['H'], [] ]  # Use empty list for no-check
check_atom_displacement = [ ['Pt'], [], [], [] ]  # Use empty list for no-check

"""
------------------------------------------------------------
| Cluster information should agree with generation setting  |
------------------------------------------------------------
"""
xyz_path = '../xyz_structures/'
comp1 = os.path.join(xyz_path, 'box32singlelayer-C3N4-surf-Pt4.xyz' )

comp2 = os.path.join(xyz_path, 'C3H8-propane.xyz')
input_molecules = [ comp1, comp2 ]
input_num_of_molecules = [1,1]

#comp2 = os.path.join(xyz_path, 'C3H7.xyz')
#comp3 = os.path.join(xyz_path, 'H.xyz')
#input_molecules = [ comp1, comp2, comp3 ]
#input_num_of_molecules = [1,1,1]

#comp2 = os.path.join(xyz_path, 'C3H6.xyz')
#comp3 = os.path.join(xyz_path, 'H.xyz')
#input_molecules = [ comp1, comp2, comp3 ]
#input_num_of_molecules = [1,1,2]

#comp2 = os.path.join(xyz_path, 'C3H6.xyz')
#comp3 = os.path.join(xyz_path, 'H2.xyz')
#input_molecules = [ comp1, comp2, comp3 ]
#input_num_of_molecules = [1,1,1]

# We just need the cluster class to use its functions/vars for the cluster components, so here we just make a fake cluster
cluster = cluster_model(input_molecules, input_num_of_molecules, ['at_position']*len(input_molecules), [()]*len(input_molecules),
                        pbc_box=(21.404922 , 24.706836 , 30.27529),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

"""
-----------------------------------------------------------------
| Perform connectivity check and/or displacement check          |
-----------------------------------------------------------------
"""
check_atom_connectivity = [c for c, n in zip(check_atom_connectivity, input_num_of_molecules) for _ in range(n)]
check_atom_displacement = [c for c, n in zip(check_atom_displacement, input_num_of_molecules) for _ in range(n)]

print( "Start cleaning..." )
traj_input, ener_input, name_input = read_trajectory( args.input , num=None )
traj_valid, ener_valid, name_valid = [],[], []

for nr, atoms in tqdm(enumerate(traj_input), total=len(traj_input), desc="Analyzing: "):
    e = ener_input[ nr ]
    m = name_input[ nr ]
    
    # First, check the energy of the structure. Skip the rest part if we have a bad E
    if e>1e3:
        continue
        
    # Second, check the neighbor connectivity changes
    check_pass = True
    index_head, index_tail = -1,-1
    for n, molecule in enumerate(cluster.templates):
        index_head = index_tail +1  # point to the first atom in mol
        index_tail = index_head +len(molecule) -1 # point to the last atom in mol
        new_mol = atoms[ index_head: index_tail+1 ]
        
        if len(check_atom_connectivity[n])>0:
            # check these atoms
            check_atoms_index = [ at.index for at in new_mol if at.symbol in check_atom_connectivity[n] ]

            connect_ref = cluster.internal_connectivity[n] # C in the input structure
            connect_ref = connect_ref[ np.ix_(check_atoms_index, check_atoms_index) ] # these rows and cols

            cutoffs = [ n for n in ngbls.natural_cutoffs(new_mol, mult=1.01) ]
            ngb_list = ngbls.NeighborList(cutoffs, self_interaction=False, bothways=True)
            ngb_list.update(new_mol)
            connect = ngb_list.get_connectivity_matrix(sparse=False) # C in the generated structure
            connect = connect[ np.ix_(check_atoms_index, check_atoms_index) ] # these rows and cols
            
            #if np.array_equal(connect, connect_ref):
            if np.sum(np.abs(connect - connect_ref))!=0:  # if change in connectivity
                check_pass = False
                break
        
        if len(check_atom_displacement[n])>0:
            check_atoms_index = [ at.index for at in new_mol if at.symbol in check_atom_displacement[n] ]
            _ , disp_length = structure_difference(new_mol[check_atoms_index], molecule[check_atoms_index])
            if np.max(disp_length) > 1.2:
                check_pass = False
                break
                
    #  Additional check with other customized conditions?
    if check_pass:
        atoms = alignment(atoms, args.align) 
        
        """
        pos = atoms.get_positions()
        L = atoms.cell.lengths()[2]
        n = [ at.index for at in atoms if at.symbol=='N' ]
        h = [ at.index for at in atoms if at.symbol=='H' ]
        pos_n = np.mean(pos[n], axis=0)[2]
        pos_h = np.mean(pos[h], axis=0)[2]
        pos_diff = pos_h - pos_n
        pos_diff = L * round(pos_diff / L)
        """

    if check_pass:
        # Final output
        traj_valid.append( atoms )
        ener_valid.append( e )
        name_valid.append( m )

# sort energy and write output
print( "Sort energy and write a new trajectory..." )
sorted_idx = np.argsort(ener_valid)
tags = None

"""
--------------------------------------------------------------------------
| If needed, we can do the similarity analysis to clean/group structures  |
--------------------------------------------------------------------------
"""
if isinstance(args.group, int):
    args.group = [ args.group ]
if args.group[0]>=0: 
    use_components_index = args.group
    use_atoms_index = [] # use these atoms for comparison
    idx_point = 0
    for im, mol in enumerate(cluster_template):
        idx = np.arange(len(mol)) + idx_point
        if cluster.global_molecule_index[im] in use_components_index: 
            use_atoms_index += idx.tolist()
        idx_point += len(mol)

    ener_ref = ener_valid[ sorted_idx[0] ]
    atoms_ref = traj_valid[ sorted_idx[0] ]
    atoms_ref = atoms_ref[use_atoms_index] # To reduce comp cost
    # Use distance matrix to measure geometry similarity?
    #dmat_ref = atoms_ref.get_all_distances(mic=True)
    
    dmat_tag, ener_tag = [], []
    for n, atoms in enumerate(traj_valid):
        # Energy
        ener_diff = ener_valid[n] - ener_ref
        ener_tag.append( ener_diff )
        # Geometry
        atoms = atoms[use_atoms_index]
        
        #dmat = atoms.get_all_distances(mic=True)
        #dmat_diff = dmat - dmat_ref
        #i, j = np.triu_indices(dmat_diff.shape[0], k=1) # Upper triangle without diagonal
        #d = dmat_diff[i,j]
        #dmat_tag.append( np.linalg.norm(d) )
        
        # Use displacement to measure geometry similarity?
        _ , d = structure_difference(atoms, atoms_ref)
        dmat_tag.append( np.mean(d) )
        
    ener_tag = np.round( ener_tag , 2 ) # if diff is within 0.01, they are the same energy (eV)

    tags =  np.transpose([ ener_tag, dmat_tag ])
    sorted_idx = np.lexsort([tags[:, i] for i in reversed(range(tags.shape[1]))]) # rank 1st item, then 2nd, etc..

"""
---------------
| FInal write |
---------------
"""
output_traj, output_ener, output_name = [],[],[]
write_seprate_lines = 1e9
with open(args.log,'w') as f1:
    if tags is None:
        header = "Index".rjust(8)+"Energy".rjust(16)+"   Name\n"
    else:
        header = "Index".rjust(8)+"Energy".rjust(16)+"E_group".rjust(8)+"XYZ_similarity".rjust(16)+"   Name\n"
    f1.write(header)
        
    for ii, i in enumerate(sorted_idx):
        if tags is None:
            line = f"{ii:8d}{ener_valid[i]:16.10g}   {name_valid[i]}\n"
        else:
            line = f"{ii:8d}{ener_valid[i]:16.10g}{tags[i][0]:8.4g}{tags[i][1]:16.10g}   {name_valid[i]}\n"
            if tags[i][0] != write_seprate_lines:
                line = "~"*20 + '\n' + line
            write_seprate_lines = tags[i][0]
        f1.write(line)
        atoms = traj_valid[i]
        atoms.translate( args.shift )

        atoms.wrap()
        
        # Write the E_group and similarity info to atoms.info
        if tags is not None:
            atoms.info["E_group"] = float(tags[i][0])
            atoms.info["XYZ_similarity"] = float(tags[i][1])
        output_traj.append( atoms )

print( "Total output frames:", len(output_traj) )
write( args.output, output_traj )
