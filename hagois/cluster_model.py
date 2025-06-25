# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:56:08 2025

@author: d2j
"""
from utility import correct_surface_normal

import numpy as np
from ase.io import read
import string
from scipy.spatial import ConvexHull


class cluster_model:
    """
    This contains the functionalities of molecular model during cluster generation
    
    molecules = list of molecules to be used. Each item is an ASE atoms obj or file
    number_of_molecules = list of int : number of molecules added. Shape = len(molecules)
    if molecules = [A,B,C] and num_of_mol is [m,n,l], the final cluster is mA+nB+lC (template)

    Rigid body packing keeps dimensionality down. 
    So we assume 6-parameter input (xyz translation and ZXZ Euler angles). 
    There may be better ways.
    
    the input X (vec) is 1D array: [x,y,z,a1,a2,a3]*m +[same thing]*n +[same]*l, dim=6m+6n+6l
    the boundary condition is 1D array: [ (hi,lo),(hi,lo),... ] with primary dim=6m+6n+6l 
    
    constraint_type = list of str: How to set bound constraints of each molecule. Shape = len(molecules)
    constraint_value = list of constraint boundary. length = len(molecules). Each item varies.
    
    """
    def __init__(self, molecules, num_of_molecules,
                 constraint_type, constraint_value,
                 pbc_box=None,
                 ):
        self.molecules = molecules
        self.num_of_molecules = num_of_molecules
        self.constraint_type = constraint_type
        self.constraint_value = constraint_value
        self.pbc_box = pbc_box
        
    # prepare the molecule (atoms obj) by adding missing info
    # Also this ensures that "molecules" is a list of ASE atoms
    def init_molelcules(self):
        for i, mol in enumerate(self.molecules):
            atoms = read(mol) # if list item is file name
            if self.pbc_box is not None:
                atoms.set_pbc( (True,True,True) )
                atoms.set_cell( self.pbc_box )
            if atoms.has('residuenames'):
                pass
            else:  # Generate a random three-letter string as name
                new_resname = [i for i in string.ascii_uppercase]
                new_resname = ''.join( np.random.choice(new_resname, 3, replace=True) )
                atoms.new_array('residuenames', [new_resname]*len(atoms), str)
            self.molecules[i] = atoms
            
    # Translate a molecule (atoms) to center and rotate it by Euler angles (degrees)
    def _move_a_molecule(self, atoms, center, angles):
        atoms.translate( center - np.mean(atoms.get_positions(), axis=0) ) # geometric center
        atoms.euler_rotate(center=center, phi=angles[0], theta=angles[1], psi=angles[2])

    # From constraint_type and _value, generate the bound condition for the algorithm.
    # Output: templates (shape= # of molecules in cluster), boundary ( 6*templates * tuple of size 2)
    # Output: go_conversion_rule (shape = templates) contains items where spherical condition is used.
    def generate_bounds(self):
        go_templates, go_boundary, go_conversion_rule = [],[], []
        for n in range(len(self.molecules)):
            # For each molecule, generate its bound ( list of 6 tuples, each tuple is (lo, hi))
            new_mol = self.molecules[n].copy()
            conversion_rule_para = ()
            
            if self.constraint_type[n] == 'at_position':
                """
                Put molecule at a position (X,Y,Z) with certain orientation (fixed) or random orientation
                if input parameter dim = 6, i.e. [Center of molecule, 3 Euler angles in degrees], then fix the molecule
                
                length of conversion_rule_para is 0
                """
                if len( self.constraint_value[n] )==6:  # fix all
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), self.constraint_value[n][3:] )
                    bound = [(0,0)]*6
                elif len( self.constraint_value[n] )==3:  # only fix the position
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] )
                    bound = [(0,0)]*3 + [(0,360)]*3
                else:
                    raise ValueError('Number of parameters should be 6 or 3')

            elif self.constraint_type[n] == 'in_box':
                """
                Put molecule within a box region defined by the lower and upper corner of the box
                Input parameter must be [xlo, ylo, zlo, xhi, yhi, zhi]
                
                length of conversion_rule_para is 0
                """
                if len( self.constraint_value[n] )==6:
                    box_size = np.array( self.constraint_value[n] )
                    box_size = box_size[3:] - box_size[:3]
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] ) # move molecule to lower corner
                    bound = [ (0,box_size[0]) , (0,box_size[1]) , (0,box_size[2]) ] + [(0,360)]*3
                else:
                    raise ValueError('Number of parameters should be 6')
                    
            elif self.constraint_type[n] == 'in_sphere_shell':
                """
                Put molecule within a sphere region defined by the center of sphere and radius along X,Y,Z directions
                Input parameter must be [X,Y,Z, R_X, R_Y, R_Z] for sphere or [X,Y,Z, R_X_out, R_Y_out, R_Z_out, delta_R] for shell (delta_R is the ratio of shell wall thickness w.r.t R_out)
                Output bound is different from the previous case since position + Euler angles does not confine a sphere region.
                Here, the position is replaced by spherical coordinates: X,Y,Z --> rho, theta, phi. 
                The question is how to differentiate the first three parameters? Once we write bound and add to go_boudary, we don't know if they are cart or sphe coordinates. 
                Also, later when vec is converted to XYZ coord, we don't know if they are cart or sphe coord.
                Work around: use another list to indicate conversion rule: if empty -> cart; if length=3 -> sphe. This could be updated by a smarter way without defining anything (maybe)...
                
                length of conversion_rule_para is 3
                """
                if len( self.constraint_value[n] )==6 or len( self.constraint_value[n] )==7:
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] ) # move molecule to the center of sphere X,Y,Z
                    conversion_rule_para = tuple(self.constraint_value[n][3:6])  # Contains the three axis of ellipsoid
                    # the range of r ( or rho ) for sphere is 0~1, but for ellipsoid, it depends on three semi-axis of ellipsoid. I don't think there are closed-form expression on this.
                    if len( self.constraint_value[n] )==6: # in_sphere
                        bound = [ (0,1), (0,360) , (0,180) ] + [(0,360)]*3  # r, theta (0*2pi), phi (0~pi),  + Euler angles
                    else:  # in_shell
                        bound = [ (1-self.constraint_value[n][-1],1), (0,360) , (0,180) ] + [(0,360)]*3  
                else:
                    raise ValueError('Number of parameters should be 6 or 7')
                
            elif self.constraint_type[n] == 'on_surface':
                """
                On surface can be used only after a substrate molecule has been defined by fixed position.
                This will compute the surface of substrate first (by Convex Hull, simplest surface), and add molecules in the second step.
                Using Convex Hull is the most common way, but we can also use Alpha Shape (concave surface) using, e.g. alphashape libirary.
                Input parameter is: 
                    int: molecular index of the substrate molecule in current go_templates
                    (float,float): Adsorption distance to the substrate surface (lo,hi)
                    int: atom index from this molecule to be on the surface adsorption point
                    int: atom index from this molecule to define the orientation of molecule
                Output is the boundary of 6 variables:
                    For binding location (3): index of face (from 0,1,2...), factor1 and factor2 for a point on this surface
                    For binding distance and angle (3) : binding distance(lo,hi), rotation angle along surf_norm axis, rotate along X
                
                length of conversion_rule_para >= 2+3
                """
                # First make sure we can generate the surface of substrate correctly
                try:
                    substrate_mol = go_templates[ self.constraint_value[n][0] ] # Substrate must have more than 3 atoms
                    # rattle substrate to ensure 3d space is occupied for convex hull computation
                    for i in range(3):
                        if np.sum(np.abs(substrate_mol.get_positions()[:,i])) == 0: # We have a perfect surface (which is possible but rare)
                            substrate_mol.rattle()
                    hull = ConvexHull(substrate_mol.get_positions())
                except:
                    raise ValueError('Substrate molecule cannot be found or hull surface cannot be defined')
                    
                # Get current molecule information. Will save them to conversion_rule
                self._move_a_molecule(new_mol, (0,0,0), [0,0,0] ) #move molecule to (0,0,0) since we don't know the exact location yet
                adsorbate_at_position  = new_mol.positions[ self.constraint_value[n][2] ]
                adsorbate_in_direction = new_mol.positions[ self.constraint_value[n][3] ] - adsorbate_at_position

                # Save the surface information of substrate for future use
                # Put together the position of three vertices and surface normal direction
                number_of_faces = len(hull.simplices)
                conversion_rule_para = [ adsorbate_at_position, adsorbate_in_direction ]
                for i in range(number_of_faces):
                    point = substrate_mol.positions[hull.simplices[i]] # XYZ of three vertices in this face
                    surf_norm = hull.equations[i, :-1] # The surface normal direction
                    surf_norm = correct_surface_normal(point[0], surf_norm, substrate_mol.get_positions()) # Make sure norm points outside
                    p = np.concatenate( (point, [ surf_norm ]) , axis=0 ) # Add the three points and surface norm of every face
                    conversion_rule_para.append( tuple(p) )  
                conversion_rule_para = tuple( conversion_rule_para ) # Length will be >2+3, depending on how many faces
                
                bound = [ (-0.499, number_of_faces-1+0.499), (0,1), (0,1) , self.constraint_value[n][1], (0,360), (0,90)]
                
            elif self.constraint_type[n] == 'replace':
                """
                Replace certain atoms in another substrate molecule defined by fixed position
                Input parameter is: 
                    int: molecualr index of the substrate molecule. (Keeping this for future flexibility)
                    tuple of int (replacement list): the atom index to be replaced in the substrate
                Output parameter (so far):
                    the index of items in the replacement list + not used*5
                
                length of conversion_rule_para is 1
                """
                conversion_rule_para = ( self.constraint_value[n][0], tuple(self.constraint_value[n][1]) ) # The mol idx, and atom idx
                num_of_site = len(self.constraint_value[n][1])

                bound = [(-0.499, num_of_site-1+0.499) , (0,0), (0,0) ] + [(0,0)]*3
            else:
                raise ValueError('Constraint type is not supported')

            go_templates       += [new_mol] *self.num_of_molecules[n] # How many molecules considered in the cluster?
            go_boundary        += bound     *self.num_of_molecules[n] # add the boundary condition for each molecule. This will be passed to the algorithm
            go_conversion_rule += [conversion_rule_para] *self.num_of_molecules[n] # if cart coord (empty) or spherical coord (len=3)
            
        return go_templates, np.array(go_boundary), go_conversion_rule
