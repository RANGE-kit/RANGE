# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:56:08 2025

@author: d2j
"""
from RANGE_go.utility import correct_surface_normal

import numpy as np

from ase.io import read
from ase import Atoms
from ase import neighborlist as ngbls

import string
from scipy.spatial import ConvexHull, Delaunay, cKDTree


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
                 pbc_box=None, pbc_applied=(True,True,True),
                 ):
        self.molecules = molecules
        self.num_of_molecules = num_of_molecules
        self.constraint_type = constraint_type
        self.constraint_value = constraint_value
        self.pbc_box = pbc_box
        self.pbc_applied = pbc_applied
        
        # prepare the molecule (atoms obj) by adding missing info
        # Also this ensures that "molecules" is a list of ASE atoms
        for i, mol in enumerate(self.molecules):
            if isinstance(mol, Atoms):  # if list item is already ASE atoms
                atoms = mol.copy()
            else: 
                try:
                    atoms = read(mol) # if list item is file name
                except:
                    raise ValueError( f"{mol} cannot be used as either a file or an ASE atoms. Check input." )
            if self.pbc_box is None:
                atoms.set_pbc( (False,False,False) )
                atoms.set_cell( (0,0,0) )
            else:
                atoms.set_pbc( self.pbc_applied )
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
    # Output: templates (shape= # of molecules in cluster), boundary ( always= 6*templates * tuple of size 2)
    # Output: go_conversion_rule (shape = templates) contains items where spherical condition is used.
    def generate_bounds(self):
        go_templates, go_boundary, go_conversion_rule = [], [], []
        self.internal_connectivity, self.global_molecule_index = [],[]
        for n in range(len(self.molecules)):
            # For each molecule, generate its bound ( list of 6 tuples, each tuple is (lo, hi))
            new_mol = self.molecules[n].copy()
            conversion_rule_para = ()
            
            if self.constraint_type[n] == 'at_position':
                """
                Put molecule at a position (X,Y,Z) with certain orientation (fixed) or random orientation
                if input parameter dim = 6, i.e. [Center of molecule, 3 Euler angles in degrees], then fix the molecule
                
                conversion_rule_para starts with at_position
                """
                if len( self.constraint_value[n] )==6:  # fix all
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), self.constraint_value[n][3:] )
                    bound = [(0,0)]*6
                elif len( self.constraint_value[n] )==3:  # only fix the position
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] )
                    bound = [(0,0)]*3 + [(0,360)]*3
                elif len( self.constraint_value[n] )==0:  # As-is
                    bound = [(0,0)]*6
                else:
                    raise ValueError('Number of parameters should be 6 or 3 or 0')
                conversion_rule_para = ('at_position',)

            elif self.constraint_type[n] == 'in_box':
                """
                Put molecule within a box region defined by the lower and upper corner of the box
                Input parameter must be [xlo, ylo, zlo, xhi, yhi, zhi]
                
                for input parameter is [xlo, ylo, zlo, xhi, yhi, zhi] *2
                The second box is used the exclude the region (outside box)
                
                conversion_rule_para starts with in_box for inside box, in_box_out for inside+outside box
                """
                if len( self.constraint_value[n] )==6: # Inside box only
                    box_size = np.array( self.constraint_value[n] )
                    box_size = box_size[3:] - box_size[:3]
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] ) # move molecule to lower corner
                    bound = [ (0,box_size[0]) , (0,box_size[1]) , (0,box_size[2]) ] + [(0,360)]*3
                    conversion_rule_para = ('in_box',)
                elif len( self.constraint_value[n] )==12: # Inside first box and outside the second box
                    box_size = np.array( self.constraint_value[n] )
                    box_size_outer = box_size[3:6] - box_size[:3]
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] ) # move molecule to lower corner
                    bound = [ (0,box_size_outer[0]) , (0,box_size_outer[1]) , (0,box_size_outer[2]) ] + [(0,360)]*3
                    box_inner_lo = box_size[6:9] - box_size[:3]
                    box_inner_hi = box_size[9: ] - box_size[:3]
                    conversion_rule_para = ('in_box_out', tuple(box_inner_lo), tuple(box_inner_hi) )
                else:
                    raise ValueError(f'Number of parameters should be 6 or 12. Current we have {len( self.constraint_value[n] )} for {self.constraint_value[n]}' )
                
            elif self.constraint_type[n] == 'in_sphere_shell':
                """
                Put molecule within a sphere region defined by the center of sphere and radius along X,Y,Z directions
                Input parameter must be [X,Y,Z, R_X, R_Y, R_Z] for sphere or [X,Y,Z, R_X_out, R_Y_out, R_Z_out, delta_R] for shell (delta_R is the ratio of shell wall thickness w.r.t R_out)
                Output bound is different from the previous case since position + Euler angles does not confine a sphere region.
                Here, the position is replaced by spherical coordinates: X,Y,Z --> rho, theta, phi. 
                The question is how to differentiate the first three parameters? Once we write bound and add to go_boudary, we don't know if they are cart or sphe coordinates. 
                Also, later when vec is converted to XYZ coord, we don't know if they are cart or sphe coord.
                Work around: use another list to indicate conversion rule: if empty -> cart; if length=3 -> sphe. This could be updated by a smarter way without defining anything (maybe)...
                
                conversion_rule_para starts with in_sphere_shell
                """
                if len( self.constraint_value[n] )==6 or len( self.constraint_value[n] )==7:
                    self._move_a_molecule(new_mol, tuple(self.constraint_value[n][:3]), [0,0,0] ) # move molecule to the center of sphere X,Y,Z
                    conversion_rule_para = ( ['in_sphere_shell']+[i for i in self.constraint_value[n][3:6]] ) # Contains the three axis of ellipsoid
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
                    For binding distance and angle (3) : binding distance(lo,hi), rotation angle along surf_norm axis, Null (rotate to surf)
                
                conversion_rule_para starts with on_surface
                """
                # First make sure we can generate the surface of substrate correctly
                try:
                    substrate_mol = go_templates[ self.constraint_value[n][0] ] 
                    assert len(substrate_mol)>3, 'Substrate must have more than 3 atoms'
                    # rattle substrate to ensure 3d space is occupied for convex hull computation
                    for i in range(3):
                        if np.sum(np.abs(substrate_mol.get_positions()[:,i])) == 0: # We have a perfect surface (which is possible but rare)
                            substrate_mol.rattle()
                    hull = ConvexHull(substrate_mol.get_positions())
                except:
                    raise ValueError('Substrate molecule cannot be found or hull surface cannot be defined')
                    
                # Get current molecule information. Will save them to conversion_rule
                self._move_a_molecule(new_mol, (0,0,0), [0,0,0] ) #move molecule to (0,0,0) since we don't know the exact location yet
                adsorbate_at_position  = self.constraint_value[n][2] # new_mol.positions[ self.constraint_value[n][2] ]
                adsorbate_in_direction = self.constraint_value[n][3] # new_mol.positions[ self.constraint_value[n][3] ] - adsorbate_at_position

                # Save the surface information of substrate for future use
                # Put together the position of three vertices and surface normal direction
                number_of_faces = len(hull.simplices) # dim >= 4
                conversion_rule_para = [ 'on_surface', adsorbate_at_position, adsorbate_in_direction ]
                for i in range(number_of_faces):
                    point = substrate_mol.positions[hull.simplices[i]] # XYZ of three vertices in this face
                    surf_norm = hull.equations[i, :-1] # The surface normal direction
                    surf_norm = correct_surface_normal(point[0], surf_norm, substrate_mol.get_positions()) # Make sure norm points outside
                    p = np.concatenate( (point, [ surf_norm ]) , axis=0 ) # Add the three points and surface norm of every face
                    conversion_rule_para.append( tuple(p) )  
                conversion_rule_para = tuple( conversion_rule_para ) # Length will be >2+3, depending on how many faces
                
                bound = [ (-0.499, number_of_faces-1+0.499), (0,1), (0,1) , self.constraint_value[n][1], (0,360), (0,0)]
              
            elif self.constraint_type[n] == 'in_pore':
                """
                Inside the pore defined by surfaces (defined by a set of atoms)
                Input parameter is:
                    int: molecular index of the substrate
                    list of int: list of atom index in the substrate to make the convex hull surface
                    float: grid point resolution (grid spacing)
                Output is:
                    index in grid points, and rotational angles
                conversion_rule_para starts with in_pore
                """
                substrate_mol = go_templates[ self.constraint_value[n][0] ] 
                points = substrate_mol.get_positions()[ np.array(self.constraint_value[n][1]) ]
                assert len(points)>3, 'Need at least 4 points'
                hull = ConvexHull( points )
                deln = Delaunay(points[hull.vertices])
                # Grid points:
                points_lo, points_hi = np.amin(points,axis=0)-0.1, np.amax(points,axis=0)+0.1 # Dim = 3
                dx = self.constraint_value[n][2] 
                x = np.arange(points_lo[0], points_hi[0], dx)
                y = np.arange(points_lo[1], points_hi[1], dx)
                z = np.arange(points_lo[2], points_hi[2], dx)
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                # Points inside the convex hull
                inside = deln.find_simplex(grid_points) >= 0
                inner_grid = grid_points[inside]
                # Remove points too close to any atom
                tree = cKDTree(points)
                dist, _ = tree.query(inner_grid, k=1)
                filtered_grid = inner_grid[dist > 1.0 ]  # min dist for cutoff =1.0. Dim = N*3
                
                # Output:
                conversion_rule_para = ( 'in_pore', tuple(filtered_grid) )
                bound = [ (-0.499, len(filtered_grid)-1+0.499), (0,0), (0,0) ] + [(0,360)]*3
                
            elif self.constraint_type[n] == 'layer': 
                """
                Define grid points uniformly on a plane surface
                Input parameter is:
                    array (Dim=3) *4, float, int, int for layer
                    three points defining the surface and a forth point defining surface norm direction
                    float gives the grid spacing. First int gives the atom id on the surface, second int gives the atom id for direction.
                Output is:
                    atom id on the surface, grid points
                conversion_rule_para starts with layer
                """
                if len( self.constraint_value[n] ) == 7: # (P1,P2,P3,P4, spacing, idx1, idx2) for plane (layer)
                    u = np.array(self.constraint_value[n][1]) - np.array(self.constraint_value[n][0])
                    v = np.array(self.constraint_value[n][2]) - np.array(self.constraint_value[n][0])
                    w = np.array(self.constraint_value[n][3]) - np.array(self.constraint_value[n][0])
                    surf_norm = np.cross(u, v)
                    if np.dot( surf_norm, w ) <0:
                        surf_norm = -surf_norm/np.linalg.norm(surf_norm)
                    else:
                        surf_norm = surf_norm/np.linalg.norm(surf_norm)
                    n_u = max(int(np.linalg.norm(u) / self.constraint_value[n][4] ) + 1, 2)
                    n_v = max(int(np.linalg.norm(v) / self.constraint_value[n][4] ) + 1, 2)
                    s = np.linspace(0, 1, n_u)
                    t = np.linspace(0, 1, n_v)
                    S, T = np.meshgrid(s, t)
                    grid_points = self.constraint_value[n][0] + (S.ravel()[:, None] * u) + (T.ravel()[:, None] * v)
                    # Align molecule to surf norm
                    p = new_mol.get_positions()
                    new_mol.translate( np.array(self.constraint_value[n][0]) - p[self.constraint_value[n][5]] )
                    new_mol.rotate(p[self.constraint_value[n][6]] - p[self.constraint_value[n][5]] , surf_norm, center=np.array(self.constraint_value[n][0]))
                    
                    conversion_rule_para = ( 'layer', self.constraint_value[n][5], tuple(grid_points) )
                else:
                    raise ValueError('Wrong number of input constraint and parameter')
                # Output:
                bound = [ (-0.499, len(grid_points)-1+0.499), (0,0), (0,0) ] + [(0,0)]*3
            
            elif self.constraint_type[n] == 'micelle':
                """
                Define grid points uniformly on an ellipsoid surface
                Input parameter is:
                    float * 3, array (Dim=3), float, int, int for ellipsoid
                    first three float give three primary axis.  Array gives the center of micelle. Last float gives grid spacing.
                    First int gives the atom id on the surface, second int gives the atom id for direction.
                Output is:
                    grid points, grid normal
                conversion_rule_para starts with micelle
                """
                if len( self.constraint_value[n] ) == 7: # (a,b,c,center, spacing, idx1, idx2) for ellipsoid (micelle)
                    a,b,c = self.constraint_value[n][0],self.constraint_value[n][1],self.constraint_value[n][2]
                    center, spacing = self.constraint_value[n][3],self.constraint_value[n][4]
                    atom_surf, atom_dir = self.constraint_value[n][5],self.constraint_value[n][6]
                    # Estimate number of divisions in theta and phi
                    n_theta = max(int(np.pi * b / spacing), 2)   # latitude divisions
                    n_phi = max(int(2 * np.pi * a / spacing), 2) # longitude divisions
                    theta_vals = np.arccos(np.linspace(1, -1, n_theta))  # from 0 to pi. Sample cos(theta) for better uniformity
                    phi_vals = np.linspace(0, 2*np.pi, n_phi)
                    theta, phi = np.meshgrid(theta_vals, phi_vals, indexing='ij')  # Create meshgrid
                    # Parametric equations
                    x = center[0] + a * np.sin(theta) * np.cos(phi)
                    y = center[1] + b * np.cos(theta)
                    z = center[2] + c * np.sin(theta) * np.sin(phi)
                    # Compute normals via gradient of implicit function
                    nx = (x - center[0]) / (a * a)
                    ny = (y - center[1]) / (b * b)
                    nz = (z - center[2]) / (c * c)
                    # Normalize normals
                    norm = np.sqrt(nx**2 + ny**2 + nz**2)
                    nx, ny, nz = nx/norm, ny/norm, nz/norm
                    # Stack points and normals
                    grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
                    grid_normals = np.column_stack([nx.ravel(), ny.ravel(), nz.ravel()])
                    
                    conversion_rule_para = ( 'micelle', atom_surf, atom_dir, tuple(grid_points), tuple(grid_normals) )
                    new_mol.translate( center - new_mol.positions[self.constraint_value[n][5]] )
                else:
                    raise ValueError('Wrong number of input constraint and parameter')
                # Output:
                bound = [ (-0.499, len(grid_points)-1+0.499), (0,0), (0,0) ] + [(0,0)]*3
                
            elif self.constraint_type[n] == 'replace':
                raise ValueError('The replace constraint has been temporarily disabled due to an unstable behavior. Fix will be implemented soon.')
                """
                Replace certain atoms in another substrate molecule (the first in order) by atom index
                Input parameter is: 
                    tuple of int (replacement list): the atom index to be replaced in the substrate
                Output parameter (so far):
                    the index of items in the replacement list + 5 tuples
                conversion_rule_para starts with replace
                """
                #conversion_rule_para = ( self.constraint_value[n][0], tuple(self.constraint_value[n][1]) ) # The mol idx, and atom idx
                conversion_rule_para = ( 'replace', tuple(self.constraint_value[n]) ) # atom idx. The mol idx will always be 0 (the first mol)
                num_of_site = len(self.constraint_value[n])
                bound = [(-0.499, num_of_site-1+0.499) , (0,0), (0,0) ] + [(0,360)]*3
                
            else:
                raise ValueError('Constraint type is not supported')

            go_templates       += [new_mol] *self.num_of_molecules[n] # How many molecules considered in the cluster?
            go_boundary        += bound     *self.num_of_molecules[n] # add the boundary condition for each molecule. This will be passed to the algorithm
            go_conversion_rule += [conversion_rule_para] *self.num_of_molecules[n] # if cart coord (empty) or spherical coord (len=3)
            
            # Connectivity
            cutoffs = [ c for c in ngbls.natural_cutoffs(new_mol, mult=1.01) ]
            ngb_list = ngbls.NeighborList(cutoffs, self_interaction=False, bothways=True)
            ngb_list.update(new_mol)
            connect = ngb_list.get_connectivity_matrix(sparse=False)
            #connect = np.triu(connect, k=1).flatten().tolist() # upper triangle without diagnol, into 1d array [1,0,0,1,0,...]
            for _ in range(self.num_of_molecules[n]):
                self.internal_connectivity.append( connect )
                self.global_molecule_index.append( n )
        self.templates = go_templates
        self.boundary = go_boundary
        self.conversion_rule = go_conversion_rule
        return go_templates, np.array(go_boundary), go_conversion_rule
    
        
