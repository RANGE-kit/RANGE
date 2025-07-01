# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:10:17 2025

@author: d2j
"""

# Utility functions
import numpy as np
import shutil
import os


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
        