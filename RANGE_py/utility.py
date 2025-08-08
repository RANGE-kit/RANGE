# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:10:17 2025

@author: d2j
"""

# Utility functions
import numpy as np

from ase.units import kJ,mol #Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from scipy.spatial.transform import Rotation 
from scipy.optimize import linear_sum_assignment


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

def select_from_diversity(X_vec, Y_ener, num_of_candidates):
    """
    Note: if X and Y are from random results, this is fine. But if from a previously calculated result,
    The lowest Y will be picked and the rest can be high Y since they have far distance from X of the lowest Y.
    Therefore, we put a Y limit (e.g. pick X from the lowest Y values to avoid picking high Y)
    """
    # Only consider the lowest several candidates
    sorted_idx = np.argsort(Y_ener)
    limit = np.amin( [len(sorted_idx), np.amax(num_of_candidates*5), int(0.2*len(sorted_idx))] )
    #Y_sorted = np.asarray(Y_ener)[sorted_idx][:limit]
    X_sorted = np.asarray(X_vec)[sorted_idx][:limit]
    
    M, D = X_sorted.shape
    N = D // 6
    # --- relative to column-wise pool average ---------------------------
    X_avg = np.mean(X_sorted, axis=0, keepdims=True)   # (1, 6*N)
    X_rel = (X_sorted - X_avg) / (X_avg + 1e-12)       # (M, 6*N)
    S = X_rel.reshape(M, N, 6)                         # (M, N, 6)    
    # pre-compute pairwise Hungarian distances
    cost = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            C = np.linalg.norm(S[i][:, None, :] - S[j][None, :, :], axis=2)
            d = C[linear_sum_assignment(C)].sum()
            cost[i, j] = cost[j, i] = d 
    # Always include lowest Y
    chosen = np.zeros(num_of_candidates, dtype=int)   # pre-allocate answer
    chosen[0] = 0       
    # greedy farthest-point selection
    chosen = np.zeros(num_of_candidates, int)
    chosen[0] = 0
    for r in range(1, num_of_candidates):
        d_min = cost[:, chosen[:r]].min(axis=1)
        d_min[chosen[:r]] = -np.inf
        chosen[r] = d_min.argmax()
    return sorted_idx[chosen]  

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
