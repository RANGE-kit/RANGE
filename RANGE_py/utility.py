# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:10:17 2025

@author: d2j
"""

# Utility functions
import numpy as np
import shutil
import os
from ase.units import kJ,mol #Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal


def cartesian_to_spherical(x, y, z):
    """Converts Cartesian (x, y, z) to spherical (rho, theta, phi)."""
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / rho)
    return rho, theta, phi

def spherical_to_cartesian(rho, theta, phi):
    """Converts spherical (rho, theta, phi) to Cartesian (x, y, z)."""
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z

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
    
def compare_two_vec_difference(pool_vec, new_vec, tol=0.01):
    v = np.copy( new_vec )
    v[v==0] = 1e-9
    diff = np.mean( np.abs((pool_vec - v)/v) , axis=1 )
    return diff #np.amin( diff )

def save_energy_summary(write_index_order, ydata, yname, output_file):
    ydata = np.round(ydata, 6)
    with open( output_file, 'w') as f1_out:
        output_line = "Index".ljust(8) + " | " + "Appear".ljust(8) + " | " + "Energy".ljust(12) + " | " + "Compute_id \n"
        f1_out.write(output_line)
        for n, idx in enumerate(write_index_order):
            output_line = f"{n:8d} | {idx:8d} | {ydata[idx]:.12g} | {yname[idx]} \n"
            f1_out.write(output_line)
            
def save_best_structure(output_directory, compute_id, save_file_path):
    job_path = os.path.join(output_directory , compute_id)
    if os.path.exists( os.path.join(job_path, 'final.xyz') ):
        shutil.copyfile( os.path.join(job_path, 'final.xyz') , save_file_path )
    elif os.path.exists( os.path.join(job_path, 'start.xyz') ):
        shutil.copyfile( os.path.join(job_path, 'start.xyz') , save_file_path )
    else:
        raise ValueError('Path ID is set to a wrong value: ', compute_id )
        

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