# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:10:17 2025

@author: d2j
"""

# Utility functions
import numpy as np

from ase import neighborlist as ngbls
from ase.units import kJ,mol #Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

from scipy.spatial.transform import Rotation 
#from scipy.spatial.distance import cdist


def cartesian_to_ellipsoidal_deg(x, y, z, A, B, C):
    x_ = x / A
    y_ = y / B
    z_ = z / C
    rho = np.sqrt(x_**2 + y_**2 + z_**2)
    # Compute azimuthal angle θ in degrees (0 to 360°)
    theta = np.degrees(np.arctan2(y_, x_))
    if theta < 0:
        theta += 360.0
    # Compute inclination angle φ in degrees (0 to 180°)
    if rho != 0:
        phi = np.degrees(np.arccos(np.clip(z_ / rho, -1.0, 1.0)))
    else:
        phi = 0.0  # arbitrary when rho = 0
    return rho, theta, phi

def ellipsoidal_to_cartesian_deg(rho, theta, phi, A, B, C):
    # Use degree-based trig functions
    sin_phi = np.sin(np.radians(phi))
    cos_phi = np.cos(np.radians(phi))
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    x = A * rho * sin_phi * cos_theta
    y = B * rho * sin_phi * sin_theta
    z = C * rho * cos_phi
    return x, y, z

def rotate_atoms_by_euler(atoms, center_of_geometry, phi, theta, psi):
    pos = atoms.get_positions() - center_of_geometry
    rot = Rotation.from_euler('ZXZ', [phi, theta, psi], degrees=True)
    Rmat = rot.as_matrix()  # Create ZXZ rotation matrix
    new_pos = pos @ Rmat.T + center_of_geometry
    atoms.set_positions(new_pos)
    return atoms

def get_translation_and_euler_from_positions(pos_start, pos_final):
    centroid_start = np.mean(pos_start, axis=0)
    centroid_final = np.mean(pos_final, axis=0)
    p_start = pos_start - centroid_start
    p_final = pos_final - centroid_final
    if np.sum(np.abs(pos_start-pos_final))>0.01:
        # Best-fit rotation matrix using Kabsch algorithm
        H = p_start.T @ p_final
        U, _, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T
        # Correct improper rotation if needed
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T
        # Extract Euler angles in ZXZ
        rot = Rotation.from_matrix(R_matrix)
        phi, theta, psi = rot.as_euler('ZXZ', degrees=True)
        # Translation vector
        rotated_centroid = R_matrix @ centroid_start
        translation_vector = centroid_final - rotated_centroid
        x, y, z = translation_vector
    else:
        x, y, z, phi, theta, psi = 0,0,0,0,0,0
    return   x, y, z, phi, theta, psi

def project_points_onto_vector(points, vector):
    v = np.array(vector)
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        raise ValueError("Vector must not be zero.")
    dot_products = np.dot(points, v)
    projections = np.outer(dot_products / v_norm_sq, v)
    return projections

def correct_surface_normal(one_surface_vertice, surf_normal, points):
    projected_points = project_points_onto_vector(points, surf_normal)
    projected_vertice = project_points_onto_vector(one_surface_vertice, surf_normal)
    v = np.mean(projected_points, axis=0) - projected_vertice
    v = v/np.linalg.norm(v) # Normalized vector pointing inside
    if np.dot( surf_normal, v.flatten() ) >0: # this should be either 1 or -1. if >0, surf_normal needs the opposite
        surf_normal = -surf_normal
    return surf_normal

def compute_differences(list_of_X, X_ref):
    diff = np.mean(np.abs( np.asarray(list_of_X) - X_ref ) , axis=1)
    return diff

def select_max_diversity(X_vec, Y_ener, num_of_candidates):
    sorted_idx = np.argsort(Y_ener)
    X_sorted = np.asarray(X_vec)[sorted_idx]
    X_sorted = X_sorted[:, ~np.all(X_sorted == X_sorted[0, :], axis=0)]  ## Remove constant columns
    X_sorted = (X_sorted - X_sorted.mean(axis=0)) / X_sorted.std(axis=0)  ## z-score normalization
    Y_sorted = np.round( np.asarray(Y_ener)[sorted_idx] , 6 )
    # Bin candidates by energy
    bin_size = np.abs(Y_sorted[0])*1E-4 + 1E-12 # 0.1% of GM
    bins = {}
    for n in range( len(Y_sorted) ):
        idx = int( Y_sorted[n] / bin_size )
        if idx in bins:
            bins[idx].append( n )
        else:
            bins[idx] = [ n ]
    keys = sorted(list(bins.keys()))
    # pick candidate from bins
    selected_indices = []
    num_cand_per_bin = np.amax( (2, int(num_of_candidates/len(keys))) )
    bin_idx = 0
    while len(selected_indices) < num_of_candidates:
        idx = bins[ keys[bin_idx] ]
        if Y_sorted[idx][-1] - Y_sorted[idx][0] < 1E-5: # Same Y in this bin
            diff = compute_differences( X_sorted[idx], X_sorted[idx[0]] )
            if np.amax(diff) > 0.1:
                selected_indices += [idx[0], idx[np.argmax(diff)]]
            else:
                selected_indices += [idx[0]]
        elif len(idx)>num_cand_per_bin:
            diff = compute_differences( X_sorted[idx], X_sorted[idx[0]] )
            selected_indices += [idx[0], idx[np.argmax(diff)]] #[ [n*cand_interval] for n in range(num_cand_per_bin) ]
        else:
            selected_indices += [idx[0]]
        bin_idx += 1
        if bin_idx==len(keys): # All keys used before having enough candidates
            selected_indices += [len(X_sorted)-i-1 for i in range(num_of_candidates-len(selected_indices))]
    selected_indices = selected_indices[:num_of_candidates]  # To avoid over-adding
    return sorted_idx[selected_indices]

def check_structure(atoms, energy, input_tuple):
    # Always distance check
    dist = atoms.get_all_distances(mic=True, vector=False)
    dist = dist[np.triu_indices(dist.shape[0], k=1)] # only upper triangle without diagonal values
    if np.amin(dist)<0.7: # bad distance. collapse.
        energy = 2E9 # bad structure
    elif input_tuple is not None:
        # input_tuple must be a tuple of size 2: 0=cluster, 1=a list of molecule id
        cluster = input_tuple[0] # cluster class to get templates, internal_connectivity and global_molecule_index
        explore_mol_index = input_tuple[1] # The molecules to check connectivity
        index_head, index_tail = -1,-1
        for n, molecule in enumerate(cluster.templates):
            index_head = index_tail +1  # point to the first atom in mol
            index_tail = index_head +len(molecule) -1 # point to the last atom in mol
            # Connectivity of this part
            if cluster.global_molecule_index[n] in explore_mol_index and index_tail>index_head:  # no need for a single atom part
                connect_ref = cluster.internal_connectivity[n]
                new_mol = atoms[ index_head: index_tail+1 ]
                cutoffs = [ n for n in ngbls.natural_cutoffs(new_mol, mult=1.01) ]
                ngb_list = ngbls.NeighborList(cutoffs, self_interaction=False, bothways=True)
                ngb_list.update(new_mol)
                connect = ngb_list.get_connectivity_matrix(sparse=False)
                #connect = np.triu(connect, k=1).flatten() # upper triangle without diagnol, into 1d array [1,0,0,1,0,...]
                if not np.array_equal(connect, connect_ref):
                    energy = 2E8
                    break
    return energy
    
# UFF force field parameter for LJ interaction. Eps in kJ/mol, Sig in Angstrom
# "UFF, a Full Periodic Table Force Field for Molecular Mechanics and Molecular Dynamics Simulations" J Am Chem Soc, 114, 10024-10035 (1992) https://doi.org/10.1021/ja00051a040
# Output is Eps (in eV) and Sig (in Ang)
def get_UFF_para(element_symbol):
    UFF_table = {
         "Ac" : [  0.138 ,  3.099 ],
         "Ag" : [  0.151 ,  2.805 ],
         "Al" : [  2.113 ,  4.008 ],
         "Am" : [  0.059 ,  3.012 ],
         "Ar" : [  0.774 ,  3.446 ],
         "As" : [  1.293 ,  3.768 ],
         "At" : [  1.188 ,  4.232 ],
         "Au" : [  0.163 ,  2.934 ],
          "B" : [  0.753 ,  3.638 ],
         "Ba" : [  1.523 ,  3.299 ],
         "Be" : [  0.356 ,  2.446 ],
         "Bi" : [  2.167 ,  3.893 ],
         "Bk" : [  0.054 ,  2.975 ],
         "Br" : [   1.05 ,  3.732 ],
          "C" : [  0.439 ,  3.431 ],
         "Ca" : [  0.996 ,  3.028 ],
         "Cd" : [  0.954 ,  2.537 ],
         "Ce" : [  0.054 ,  3.168 ],
         "Cf" : [  0.054 ,  2.952 ],
         "Cl" : [   0.95 ,  3.516 ],
         "Cm" : [  0.054 ,  2.963 ],
         "Co" : [  0.059 ,  2.559 ],
         "Cr" : [  0.063 ,  2.693 ],
         "Cs" : [  0.188 ,  4.024 ],
         "Cu" : [  0.021 ,  3.114 ],
         "Dy" : [  0.029 ,  3.054 ],
         "Er" : [  0.029 ,  3.021 ],
         "Es" : [   0.05 ,  2.939 ],
         "Eu" : [  0.033 ,  3.112 ],
          "F" : [  0.209 ,  2.997 ],
         "Fe" : [  0.054 ,  2.594 ],
         "Fm" : [   0.05 ,  2.927 ],
         "Fr" : [  0.209 ,  4.365 ],
         "Ga" : [  1.736 ,  3.905 ],
         "Gd" : [  0.038 ,  3.001 ],
         "Ge" : [  1.586 ,  3.813 ],
          "H" : [  0.184 ,  2.571 ],
         "He" : [  0.234 ,  2.104 ],
         "Hf" : [  0.301 ,  2.798 ],
         "Hg" : [  1.611 ,   2.41 ],
         "Ho" : [  0.029 ,  3.037 ],
          "I" : [  1.418 ,  4.009 ],
         "In" : [  2.506 ,  3.976 ],
         "Ir" : [  0.305 ,   2.53 ],
          "K" : [  0.146 ,  3.396 ],
         "Kr" : [   0.92 ,  3.689 ],
         "La" : [  0.071 ,  3.138 ],
         "Li" : [  0.105 ,  2.184 ],
         "Lu" : [  0.172 ,  3.243 ],
         "Lw" : [  0.046 ,  2.883 ],
         "Md" : [  0.046 ,  2.917 ],
         "Mg" : [  0.464 ,  2.691 ],
         "Mn" : [  0.054 ,  2.638 ],
         "Mo" : [  0.234 ,  2.719 ],
          "N" : [  0.289 ,  3.261 ],
         "Na" : [  0.126 ,  2.658 ],
         "Nb" : [  0.247 ,   2.82 ],
         "Nd" : [  0.042 ,  3.185 ],
         "Ni" : [  0.063 ,  2.525 ],
         "No" : [  0.046 ,  2.894 ],
         "Np" : [  0.079 ,   3.05 ],
          "O" : [  0.251 ,  3.118 ],
         "Os" : [  0.155 ,   2.78 ],
          "P" : [  1.276 ,  3.695 ],
         "Pa" : [  0.092 ,   3.05 ],
         "Pb" : [  2.774 ,  3.828 ],
         "Pd" : [  0.201 ,  2.583 ],
         "Pm" : [  0.038 ,   3.16 ],
         "Po" : [   1.36 ,  4.195 ],
         "Pr" : [  0.042 ,  3.213 ],
         "Pt" : [  0.335 ,  2.454 ],
         "Pu" : [  0.067 ,   3.05 ],
         "Ra" : [   1.69 ,  3.276 ],
         "Rb" : [  0.167 ,  3.665 ],
         "Re" : [  0.276 ,  2.632 ],
         "Rh" : [  0.222 ,  2.609 ],
         "Rn" : [  1.038 ,  4.245 ],
         "Ru" : [  0.234 ,   2.64 ],
          "S" : [  1.146 ,  3.595 ],
         "Sb" : [  1.879 ,  3.938 ],
         "Sc" : [  0.079 ,  2.936 ],
         "Se" : [  1.218 ,  3.746 ],
         "Si" : [  1.682 ,  3.826 ],
         "Sm" : [  0.033 ,  3.136 ],
         "Sn" : [  2.372 ,  3.913 ],
         "Sr" : [  0.983 ,  3.244 ],
         "Ta" : [  0.339 ,  2.824 ],
         "Tb" : [  0.029 ,  3.074 ],
         "Tc" : [  0.201 ,  2.671 ],
         "Te" : [  1.665 ,  3.982 ],
         "Th" : [  0.109 ,  3.025 ],
         "Ti" : [  0.071 ,  2.829 ],
         "Tl" : [  2.845 ,  3.873 ],
         "Tm" : [  0.025 ,  3.006 ],
          "U" : [  0.092 ,  3.025 ],
          "V" : [  0.067 ,  2.801 ],
          "W" : [   0.28 ,  2.734 ],
         "Xe" : [  1.389 ,  3.924 ],
          "Y" : [  0.301 ,   2.98 ],
         "Yb" : [  0.954 ,  2.989 ],
         "Zn" : [  0.519 ,  2.462 ],
         "Zr" : [  0.289 ,  2.783 ],
         "X" :  [    0.0 ,    0.0 ],
    }
    eps = UFF_table[element_symbol][0]*(kJ/mol)
    sig = UFF_table[element_symbol][1]
    return eps, sig
