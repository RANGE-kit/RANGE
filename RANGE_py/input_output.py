# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:08:27 2025

@author: d2j
"""
import shutil
import os
import numpy as np

from ase.io import read, write
from ase.db import connect
from RANGE_py.utility import select_from_diversity
            
                   
def clean_directory(dir_path):
    for entry in os.scandir(dir_path):
        if entry.is_file() or entry.is_symlink():
            os.remove(entry.path)
        elif entry.is_dir():
            shutil.rmtree(entry.path)
        
def save_structure_to_db(atoms, vector, energy, name, db_path, **kwargs):
    """
    vector: 1d array. The generated vector (X), after coarse opt if used.
    energy: float.    The generated energy (Y)
    name: str.        The name of this structure
    **kwargs : dict     Additional metadata to store
    """
    if atoms is not None and db_path is not None:
        db = connect(db_path)
        data_added = dict(input_vector=vector, output_energy=energy, compute_name=name)
        db.write(atoms, data=data_added, **kwargs)

def read_structure_from_db( db_path, selection_strategy, num_of_strutures ):
    db = connect(db_path)
    vec, ener, name = [],[],[]
    for row in db.select():
        vec.append( row.data.input_vector )
        ener.append( row.data.output_energy )
        name.append( row.data.compute_name )
    length_of_current_pool = len(name)
    vec, ener, name = select_vector_and_energy(vec, ener, name, selection_strategy, num_of_strutures) 
    return  vec, ener, name, length_of_current_pool

def read_structure_from_directory( directory_path, selection_strategy, num_of_strutures ):
    vec, ener, name = [],[],[]
    # Get job directories and loop jobs to get vector and energy
    with os.scandir(directory_path) as entries:
        for job in entries:
            if job.is_dir():
                v = np.loadtxt( os.path.join(job.path, 'vec.txt') ) 
                e = np.loadtxt( os.path.join(job.path, 'energy.txt') ) 
                vec.append( v )
                ener.append( e )
                name.append( job.path )
    length_of_current_pool = len(name)
    vec, ener, name = select_vector_and_energy(vec, ener, name, selection_strategy, num_of_strutures)
    return  vec, ener, name, length_of_current_pool

def select_vector_and_energy(vector,energy,names, selection_strategy, num_of_strutures):
    if selection_strategy=='all' or selection_strategy==None or num_of_strutures==len(energy): # All data
        idx = np.arange(len(energy))
    elif selection_strategy == 'lowest': # select from the lowest structure with high diversity
        idx = select_from_diversity(vector, energy, num_of_strutures)
    elif selection_strategy == 'random':
        idx = np.random.choice(np.arange(len(energy)), size=num_of_strutures, replace=False) 
    else:
        try:
            idx = np.array(selection_strategy, dtype=int)
            assert len(idx)==num_of_strutures
        except:
            raise ValueError('Selection from existing pool cannot be done')
    energy = np.array(energy)[idx]
    vector = np.array(vector)[idx]  
    names = np.array(names)[idx]  
    return vector, energy, names

def convert_directory_to_db(directory_path, db_path):
    with os.scandir(directory_path) as entries:
        for job in entries:
            if job.is_dir():
                v = np.loadtxt( os.path.join(job.path, 'vec.txt') ) 
                e = np.loadtxt( os.path.join(job.path, 'energy.txt') ) 
                a = read( os.path.join(job.path, 'final.xyz') )
                m = job.path
                save_structure_to_db(a, v, e, m, db_path)

def get_CP2K_run_info(CP2K_input_script_file, initial_xyz):
    with open(CP2K_input_script_file, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            if '@set RUNTYPE' in line:
                run_type = line.split()[2]
            if '@set FNAME' in line:
                xyz_name = line.split()[2]
    if run_type=='ENERGY':
        atoms = read(initial_xyz)
    elif run_type=='GEO_OPT':
        with os.scandir('./') as entries:
            for entry in entries:
                if entry.is_file() and xyz_name in entry and '.xyz' in entry:
                    atoms = read(entry, index='-1') # The last frame of traj
    return atoms

def save_energy_summary(output_file='energy_summary.log', 
                        db_path='structure_pool.db', 
                        directory_path='results'):
    # Search data
    if os.path.exists( db_path ): # Method 1: use .db
        vec, energy, name, previous_size = read_structure_from_db( db_path, 'all', None )
        use_source = 'database'
    elif os.path.exists( directory_path ): # Method 2: use directory
        vec, energy, name, previous_size = read_structure_from_directory( directory_path, 'all', None )
        use_source = 'directory'
    else:
        raise ValueError('No result is found' )
    print('Read data from ', use_source)
    
    # Sort energy and write summary file
    sorted_idx = np.argsort(energy)
    energy = np.round(energy, 6)
    ranked_idxs, appear_idxs, ranked_energies, iteration_idxs, operator_types, full_name = [],[],[],[],[],[]
    with open( output_file, 'w') as f1_out:
        output_line = "Index".rjust(8) + "Appear".rjust(8) + "Energy".rjust(16)  
        output_line += "Iteration".rjust(10) + "Operation".rjust(10) + " Full_ID".ljust(30)+" \n"
        f1_out.write(output_line)
        for n, idx in enumerate(sorted_idx):
            m = name[idx].split('_')
            appear_idx = int(m[1])
            iteration_idx = int(m[3])
            operator_type = m[4].upper()
            output_line = f"{n:8d}{appear_idx:8d}{energy[idx]:16.10g}"
            output_line += f"{iteration_idx:10d}{operator_type:>10} {name[idx]}\n"
            f1_out.write(output_line)
            # Keep for future use
            ranked_idxs.append(n)
            appear_idxs.append(appear_idx)
            ranked_energies.append(energy[idx])
            iteration_idxs.append(iteration_idx)
            operator_types.append(operator_type)  
            full_name.append(name[idx])
    data_dict = {'ranked_id': np.array(ranked_idxs, dtype=int), 
                 'appear_id': np.array(appear_idxs, dtype=int), 
                 'ranked_ener': np.array(ranked_energies, dtype=float), 
                 'unranked_ener': energy, 
                 'iter_id': np.array(iteration_idxs, dtype=int), 
                 'op_type': np.array(operator_types, dtype=str),
                 'ranked_full_name': np.array(full_name, dtype=str),
                 }
    # Get the GM structure
    gm_id = full_name[0]
    print(gm_id)
    if use_source == 'database':
        db = connect(db_path)
        for row in db.select():
            if row.data.get("compute_name") == gm_id:
                atoms = row.toatoms()
                write('best.xyz', atoms)
                break
    elif use_source == 'directory':
        ##dir_path = os.path.join(directory_path,gm_id)
        shutil.copyfile( os.path.join(gm_id, 'final.xyz') , 'best.xyz' )   
    return data_dict

def print_code_info(print_location):
    if print_location=='Header':
        print('RANGE search starts')
    else:
        print(str(print_location))