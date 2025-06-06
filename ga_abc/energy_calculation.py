# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:56:35 2025

@author: d2j
"""

import numpy as np
#from scipy.spatial.transform import Rotation 
import os

from ase import Atoms
from ase.optimize import BFGS
#from ase.io.trajectory import Trajectory
from ase.io import write


class energy_computation:
    """
    This contains the functionalities of computing the energy of a molecular structure
    It needs the template of the cluster (generated from generate_bounds), just to know what we are modeling.
    """
    def __init__(self, templates, go_conversion_rule, 
                 calculator, calculator_type, geo_opt_para,
                 save_output_header='model',
                 ):
        """
        if calc_type == 'internal', use ASE calculator. Then calculator = ASE calculator.
        if calc_type == 'external', use external calculator. Then calculator = command line to run the external computation.
        """
        self.templates = templates
        self.go_conversion_rule = go_conversion_rule
        self.calculator = calculator
        self.calculator_type = calculator_type
        self.geo_opt_para = geo_opt_para
        self.save_output_header = save_output_header

    # Convert a vec X to 3D structure using template 
    def vector_to_cluster(self, vec):
        """
        vec : (6*templates  real-valued array, to inform the movement of molecules. 
             This is obtained from the ABC algorithm.
        templates : list of ase.Atoms (one per molecule)
        Returns an ase.Atoms super-molecule positioned & rotated.
        """
        placed, resname_of_placed, resid_of_placed = Atoms(), [], []
        for i, mol in enumerate(self.templates):
            # We need to keep molecules in templates unchanged
            m = mol.copy()
            center_of_geometry = np.mean(m.get_positions(),axis=0)

            if len(self.go_conversion_rule[i])==0: 
                t = vec[6*i : 6*i+3]  # Cartesian coord in vec by default
                euler = vec[6*i+3 : 6*i+6]  # Second 3 values are rotation Euler angles
                m.euler_rotate(center=center_of_geometry, phi=euler[0], theta=euler[1], psi=euler[2])
                m.translate( t )

            # Check if we have sphere coord in vec. If so, we have 3 semi-axis values here and need to convert t (r, theta, phi) to cart coord
            elif len(self.go_conversion_rule[i])==3: 
                t = vec[6*i : 6*i+3] 
                x = t[0] * np.sin(t[2]*np.pi/180.) * np.cos(t[1]*np.pi/180.) * self.go_conversion_rule[i][0]
                y = t[0] * np.sin(t[2]*np.pi/180.) * np.sin(t[1]*np.pi/180.) * self.go_conversion_rule[i][1]
                z = t[0] * np.cos(t[2]*np.pi/180.) * self.go_conversion_rule[i][2]
                t = np.array([x,y,z])
                euler = vec[6*i+3 : 6*i+6]  # Second 3 values are rotation Euler angles
                m.euler_rotate(center=center_of_geometry, phi=euler[0], theta=euler[1], psi=euler[2])
                m.translate( t )
                
            # Check if we have the face information from on_surface. If so, we have 3 surface info: face ID, factor 1 and 2 to look for points in plane
            # We also have the 3 rotation info to determine the orientation of molecule adsorbed on surface.
            elif len(self.go_conversion_rule[i])>4: 
                face = self.go_conversion_rule[i][ 2+round(t[0]) ] # get the specific face info: 3 points + 1 surface adsorption normal point
                adsorb_location = (face[1] - face[0])*t[1] +face[0]
                adsorb_location += (face[2] - adsorb_location)*t[2]  + face[3]*t[3]
                m.translate( adsorb_location - self.go_conversion_rule[i][0] )  # 0 = adsorbate_at_position
                m.rotate( self.go_conversion_rule[i][1], face[3], center=adsorb_location)  # 1 = adsorbate_in_direction --> surf_norm 
                m.rotate( t[4], face[3], center=adsorb_location )
                m.rotate( t[5], 'x', center=adsorb_location )
            else:
                raise ValueError(f'go_conversion_rule has a strange length: {len(self.go_conversion_rule[i])}')

            # Move and rotate the molecule to final destination
            # Or new_mol_xyz = mol.get_positions().dot(rot.T) + t where Rotation.from_euler('zxz', euler).as_matrix()
            placed += m
            resname_of_placed += m.get_array('residuenames' ).tolist()
            resid_of_placed += [i]*len(m) 
                                     
        cluster = Atoms(placed)  # concatenate
        cluster.new_array('residuenames', resname_of_placed, str)
        cluster.new_array('residuenumbers', resid_of_placed, str)
        
        return cluster

    
    # The obj func for energy computing
    # Assign possible ways to compute energy
    def obj_func_compute_energy(self, vec, computing_id, save_output_directory):
        atoms = self.vector_to_cluster(vec)
        
        # To use ASE calculator
        if self.calculator_type == 'ase': 
            atoms.calc = self.calculator
            # If anything happens (e.g. SCF not converged due to bad structure), return a fake high energy
            if self.geo_opt_para is not None:
                try:
                    dyn = BFGS(atoms)
                    dyn.run(fmax=self.geo_opt_para['fmax'], steps=self.geo_opt_para['steps'])
                    #traj = Trajectory( os.path.join(save_output_directory,'geoopt.traj'), 'w', atoms)
                    #dyn.attach(traj.write, interval=10)
                    energy = atoms.get_potential_energy()
                except:
                    energy = 1e6
            else:
                try:
                    energy = atoms.get_potential_energy()
                except:
                    energy = 1e6
                    
            new_cumpute_directory = os.path.join(save_output_directory,computing_id)
            os.makedirs( new_cumpute_directory, exist_ok=False)        
            write( os.path.join(new_cumpute_directory, 'final.xyz'), atoms )
            np.savetxt(os.path.join(new_cumpute_directory, 'vec.txt'), vec, delimiter=',')
                
        elif self.calculator_type == 'external': # To use external command
            energy = self.call_external_calculation(atoms, self.calculator , self.geo_opt_para)
            
        elif self.calculator_type == 'structural': # For structure generation
            energy = 0.0
            
        return energy
    
        
    # Call the external tool to compute energy
    def call_external_calculation(atoms, calculator_lines , geo_opt_para_line ):
        if calculator_lines == 'gfn2':
            if geo_opt_para_line is not None:
                energy = 0
            else:
                energy = 1
        else:
            raise ValueError('External calculation setting has wrong values')
        return energy 
    