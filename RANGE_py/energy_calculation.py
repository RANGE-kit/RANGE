# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:56:35 2025

@author: d2j
"""

import numpy as np
import os

from ase import Atoms
from ase.optimize import BFGS
#from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes

import subprocess
import shutil
from itertools import combinations
import time

from RANGE_py.utility import get_UFF_para, ellipsoidal_to_cartesian_deg, cartesian_to_ellipsoidal_deg, rotate_atoms_by_euler, get_translation_and_euler_from_positions
from RANGE_py.input_output import get_CP2K_run_info


"""
To replace the push apart function in previous versions, we use a rigid body optimizer
that uses LJ to compute external DOF and freeze internal DOF
"""
class RigidLJQ_calculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, templates, charge=0, epsilon='UFF', sigma='UFF', cutoff=1.0, 
                 coulomb_const=14.3996, # eV·Å/e² (vacuum permittivity included)
                 **kwargs):
        """
        Use function form: u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )
        templates: template of the cluster (generated from generate_bounds), to know individual molecules 
            List of lists. Each sublist contains atom indices of one rigid molecule.
        epsilon, sigma: LJ parameters. 
            If single value, applies to all atoms. If dict, lookup table for element symbol -> value
            
        Note: This is not applicable for replace (where molecules are removed)
        """
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.lower_cutoff = 0.3 # To avoid too close atoms
        self.coulomb_const = coulomb_const

        # Create lookup for quick atom-to-molecule membership check
        natom = 0 # index pointer
        chemical_symbol_list = []
        self.atom_to_mol = {}
        self.mol_to_atom = {}
        for i, mol in enumerate(templates): # loop all molecules
            chemical_symbol_list += mol.get_chemical_symbols()
            atom_index_in_this_mol = []
            for idx in mol: # loop all atoms in this molecule
                self.atom_to_mol[natom] = i  # This atom belongs to this mol
                atom_index_in_this_mol.append( natom )
                natom += 1
            self.mol_to_atom[i] = atom_index_in_this_mol
                
        # Make sure LJ parameter is atom-specific
        if isinstance(epsilon, (int, float)): # same value for all 
            self.epsilon = [epsilon]*natom
        elif isinstance(epsilon, dict): 
            self.epsilon = [ epsilon[i] for i in chemical_symbol_list ] # User provided
        elif epsilon=='UFF':
            self.epsilon = [ get_UFF_para(i)[0] for i in chemical_symbol_list ] # Use UFF table
        else:
            raise ValueError('Epsilon is not assigned properly')

        if isinstance(sigma, (int, float)): 
            self.sigma = [sigma]*natom
        elif isinstance(sigma, dict): 
            self.sigma = [ sigma[i] for i in chemical_symbol_list ]
        elif sigma=='UFF':
            self.sigma = [ get_UFF_para(i)[1] for i in chemical_symbol_list ]
        else:
            raise ValueError('Sigma is not assigned properly')

        # Atomic charge is also atom-specific
        if isinstance(charge, (list,np.ndarray)):
            if len(charge)!=natom: 
                raise ValueError('Charge and molecules should have the same length: ', len(charge), natom)
            else:
                self.charge = np.array(charge, dtype=float)
        elif isinstance(charge, (int, float)):
            self.charge = np.array([charge]*natom)
        else:
            raise ValueError('Charge is not assigned properly')

    def convert_force_to_rigid(self, all_positions, all_forces, dict_mol_to_atom):
        new_forces = np.zeros_like(all_forces)
        for mol in dict_mol_to_atom.values():
            mol = np.array(mol)
            pos = all_positions[mol]
            pos_center  = np.mean(pos, axis=0)
            frc = all_forces[mol]
            total_force = np.sum(frc, axis=0) 

            """
            r_rel = pos - pos_center  # Dsiplacement from center
            torque = np.sum(np.cross(r_rel, frc), axis=0)  # Torque
            # Inertia tensor
            Inertia = np.zeros((3, 3))
            for i in range(len(mol)):
                Inertia += (np.dot(r_rel[i], r_rel[i]) * np.identity(3) - np.outer(r_rel[i], r_rel[i]))
            # Compute angular velocity (safe pseudo-inverse)
            omega = np.linalg.pinv(Inertia) @ torque
            """
            
            # Reconstruct rigid-body forces
            for i, atom_idx in enumerate(mol):
                f_trans = total_force/len(mol)
                #f_rot = np.cross(omega, r_rel[i])
                new_forces[atom_idx] = f_trans #+ f_rot
        return new_forces        
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        natoms = len(positions)
        forces = np.zeros_like(positions)
        energy = 0.0

        # Loop over unique atom pairs
        for i, j in combinations(range(natoms), 2):
            # Skip if in the same molecule (rigid body constraint)
            if self.atom_to_mol[i] == self.atom_to_mol[j]:
                continue
                
            #rij = positions[j] - positions[i]
            rij = atoms.get_distance(i, j, mic=True, vector=True)
            r = np.linalg.norm(rij)
            if r > self.cutoff:
                continue
            elif r < self.lower_cutoff:
                if r<1e-5: # This helps to kick atoms when they exactly overlap.
                    rij = np.random.rand(3)
                    rij = rij/np.linalg.norm(rij)* self.lower_cutoff # vec with 0.2 length
                r = self.lower_cutoff # To avoid divided by zero when too close. 
                    
            # Lennard-Jones Potential with Lorentz-Berthelot (LB)
            sig = (self.sigma[i] + self.sigma[j] )/2
            eps = np.sqrt(self.epsilon[i]*self.epsilon[j])
            e_cutoff = 4 * eps * ((sig / self.cutoff) ** 12 - (sig / self.cutoff) ** 6) # energy at cutoff
            
            sr6 = (sig / r) ** 6
            sr12 = sr6 * sr6
            e_lj = 4*eps*(sr12 - sr6) - e_cutoff # Shift by cutoff energy to ensure continuous energy, although force will not be
            # Force magnitude and direction
            f_lj = -24 * eps / r * (2 * sr12 - sr6) # note the minus sign
            f_lj = f_lj * (rij / r)
            
            # Coulomb e and f
            q1,q2 = self.charge[i], self.charge[j]
            e_coul = self.coulomb_const * q1 * q2 / r
            f_coul = self.coulomb_const * q1 * q2 / (r ** 2)
            f_coul = f_coul * (rij / r)
            
            # Apply force (Newton's 3rd law)
            forces[i] += (f_lj + f_coul)
            forces[j] -= (f_lj + f_coul)

            energy += (e_lj + e_coul)
            #print( e_lj , e_coul )

        # Now, atoms in the same molecule need to have the same force
        forces = self.convert_force_to_rigid( positions, forces, self.mol_to_atom )
        
        self.results['energy'] = energy 
        self.results['forces'] = forces


class energy_computation:
    """
    This contains the functionalities of computing the energy of a molecular structure
    It needs the template of the cluster (generated from generate_bounds), just to know what we are modeling.
    """
    def __init__(self, templates, go_conversion_rule, 
                 calculator, calculator_type, geo_opt_para, 
                 if_coarse_calc=False, coarse_calc_para=None,
                 save_output_level = 'Full',
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
        self.save_output_level = save_output_level
        
        self.if_coarse_calc = if_coarse_calc
        if if_coarse_calc:
            try: 
                self.coarse_calc_eps = coarse_calc_para['coarse_calc_eps']
                self.coarse_calc_sig = coarse_calc_para['coarse_calc_sig']
                self.coarse_calc_chg = coarse_calc_para['coarse_calc_chg']
                self.coarse_calc_step = coarse_calc_para['coarse_calc_step']
                self.coarse_calc_fmax = coarse_calc_para['coarse_calc_fmax']
                self.coarse_calc_constraint = coarse_calc_para['coarse_calc_constraint']
            except:
                raise ValueError('Coarse optimization parameter is missing or incorrect')
            self.coarse_calculator = RigidLJQ_calculator(self.templates, charge=self.coarse_calc_chg, 
                                                         epsilon=self.coarse_calc_eps, sigma=self.coarse_calc_sig,
                                                         )

    # Convert a vec X to 3D structure using template 
    def vector_to_cluster(self, vec):
        """
        vec : (6*templates  real-valued array, to inform the movement of molecules. 
             This is obtained from the ABC algorithm.
        templates : list of ase.Atoms (one per molecule)
        Returns an ase.Atoms super-molecule positioned & rotated.
        """
        if np.any(self.templates[0].get_pbc()):
            Cell = self.templates[0].get_cell()
        else:
            Cell = None
            
        placed, resname_of_placed, resid_of_placed = Atoms(), [], []
        for i, mol in enumerate(self.templates):
            # We need to keep molecules in templates unchanged
            m = mol.copy()
            center_of_geometry = np.mean(m.get_positions(),axis=0)

            if len(self.go_conversion_rule[i])==0: 
                t = vec[6*i : 6*i+3]  # Cartesian coord in vec by default
                euler = vec[6*i+3 : 6*i+6]  # Second 3 values are rotation Euler angles
                m = rotate_atoms_by_euler(m, center_of_geometry, euler[0], euler[1], euler[2] )
                m.translate( t )
                
            # The replacement function to replace atoms. If the atom symbol is X, it means a vaccancy.
            elif len(self.go_conversion_rule[i])==1: 
                t = vec[6*i : 6*i+6]
                substrate_mol = placed.copy() #[ self.go_conversion_rule[i][0] ] # ase obj: mol to be changed
                substrate_atom_id = self.go_conversion_rule[i][0][ round(t[0]) ] # int: atom id to be changed
                symbol_new_atom = m.get_chemical_symbols()[0]
                if substrate_mol.symbols[ substrate_atom_id ] != symbol_new_atom:
                    substrate_mol.symbols[ substrate_atom_id ] = 'X'
                    m.positions[0] = substrate_mol.get_positions()[ substrate_atom_id ]
                else:  # duplicated site, then pick a new site
                    idx_avail = [k for k in self.go_conversion_rule[i][1] if substrate_mol.symbols[k]!=symbol_new_atom ]
                    idx_avail = np.random.choice(idx_avail)
                    substrate_mol.symbols[ idx_avail ] = 'X'
                    m.positions[0] = substrate_mol.get_positions()[ idx_avail ]                    
                #placed[ self.go_conversion_rule[i][0] ] = substrate_mol
                placed = substrate_mol.copy()
                #m = Atoms()
                
            # inside box with outside limit: where the two parameter inside tells the lo and hi of inner box
            elif len(self.go_conversion_rule[i])==2:     
                t = vec[6*i : 6*i+3]  # Cartesian coord in vec by default
                euler = vec[6*i+3 : 6*i+6]  # Second 3 values are rotation Euler angles
                m = rotate_atoms_by_euler(m, center_of_geometry, euler[0], euler[1], euler[2] )
                # Now we need to adjust the translation vector t
                xlo, ylo, zlo = self.go_conversion_rule[i][0]
                xhi, yhi, zhi = self.go_conversion_rule[i][1]
                if_inside_inner = (xlo <= t[0] <= xhi) and (ylo <= t[1] <= yhi) and (zlo <= t[2] <= zhi)
                if if_inside_inner:
                    distances = {'xlo': abs(t[0] - xlo), 'xhi': abs(t[0] - xhi),  
                                 'ylo': abs(t[1] - ylo), 'yhi': abs(t[1] - yhi),
                                 'zlo': abs(t[2] - zlo), 'zhi': abs(t[2] - zhi),
                                 }
                    closest_face = min(distances, key=distances.get) # Get key for the smallest distance
                    if closest_face == 'xlo':  
                        t[0] = xlo - 1e-3
                    elif closest_face == 'xhi':
                        t[0] = xhi + 1e-3
                    elif closest_face == 'ylo':
                        t[1] = ylo - 1e-3
                    elif closest_face == 'yhi':
                        t[1] = yhi + 1e-3
                    elif closest_face == 'zlo':
                        t[2] = zlo - 1e-3
                    elif closest_face == 'zhi':
                        t[2] = zhi + 1e-3
                m.translate( t )
                
            # Check if we have sphere coord in vec. If so, we have 3 semi-axis values here and need to convert t (r, theta, phi) to cart coord
            elif len(self.go_conversion_rule[i])==3: 
                t = vec[6*i : 6*i+3] 
                x,y,z = ellipsoidal_to_cartesian_deg(t[0], t[1], t[2], 
                                             self.go_conversion_rule[i][0], 
                                             self.go_conversion_rule[i][1], 
                                             self.go_conversion_rule[i][2])
                t = np.array([x,y,z])
                euler = vec[6*i+3 : 6*i+6]  # Second 3 values are rotation Euler angles
                m = rotate_atoms_by_euler(m, center_of_geometry, euler[0], euler[1], euler[2] )
                m.translate( t )
                
            # Check if we have the face information from on_surface. If so, we have 3 surface info: face ID, factor 1 and 2 to look for points in plane
            # We also have the 3 or 2 rotation info to determine the orientation of molecule adsorbed on surface.
            elif len(self.go_conversion_rule[i])>4:
                t = vec[6*i : 6*i+6] 
                face = self.go_conversion_rule[i][ 2+round(t[0]) ] # get the specific face info: 3 points + 1 surface adsorption normal point
                adsorb_surf_vector = (face[1] - face[0])*t[1] +face[0]
                adsorb_surf_vector += (face[2] - adsorb_surf_vector)*t[2]  
                adsorb_location = adsorb_surf_vector + face[3]*t[3]
                
                template_adsorb_atom_location = m.positions[self.go_conversion_rule[i][0]] # 0 = adsorbate_at_position
                template_adsorb_direction = m.positions[self.go_conversion_rule[i][1]] - template_adsorb_atom_location # 1 = direction atom
                
                m.translate( adsorb_location - template_adsorb_atom_location )  
                m.rotate( template_adsorb_direction, face[3], center=adsorb_location)  # adsorbate_in_direction --> surf_norm 
                m.rotate( t[4], face[3], center=adsorb_location )
                #m.rotate( t[5], 'x', center=adsorb_location )
            else:
                raise ValueError(f'go_conversion_rule has a strange length: {len(self.go_conversion_rule[i])}')

            # Move and rotate the molecule to final destination
            # Or new_mol_xyz = mol.get_positions().dot(rot.T) + t where Rotation.from_euler('zxz', euler).as_matrix()
            placed += m
            resname_of_placed += m.get_array('residuenames' ).tolist()
            resid_of_placed += [i]*len(m) 
                    
        cluster = Atoms(placed) #concatenate Atoms(placed)
        cluster.new_array('residuenames', resname_of_placed, str)
        cluster.new_array('residuenumbers', resid_of_placed, str)
        
        # To support vaccancy function
        del cluster[[atom.index for atom in cluster if atom.symbol=='X']]
        
        # Add PBC if needed
        if Cell is not None:
            cluster.set_pbc( (True,True,True) )
            cluster.set_cell( Cell )
            
        return cluster


    def cluster_to_vector(self, cluster, vec):
        """
        Convert cluster (ASE atoms) to vector. 
        """
        atom_idx_header = 0 # to point to the atoms in cluster
        vec_new = []
        for i, mol_old in enumerate(self.templates):
            mol_new = cluster[ atom_idx_header: atom_idx_header+len(mol_old) ]
            
            atom_idx_header += len(mol_old) # Move to the next molecule
            # positions of initial and final states
            pos_old = mol_old.get_positions()
            pos_new = mol_new.get_positions()
            x, y, z, phi, theta, psi = get_translation_and_euler_from_positions(pos_old, pos_new)
            
            # Now convert x,y,z, phi, theta, psi to proper format
            if len(self.go_conversion_rule[i])==0 or len(self.go_conversion_rule[i])==2: # position, inbox, outside inbox
                vec_new += [x,y,z, phi, theta, psi] 
                
            elif len(self.go_conversion_rule[i])==1:  # Replace function needs no revision
                vec_new += list(vec[6*i : 6*i+3])
                
            elif len(self.go_conversion_rule[i])==3: # Spherical coord
                r_trans, theta_trans, phi_trans = cartesian_to_ellipsoidal_deg(x,y,z,
                                                                               self.go_conversion_rule[i][0],
                                                                               self.go_conversion_rule[i][1],
                                                                               self.go_conversion_rule[i][2])
                vec_new += [r_trans, theta_trans, phi_trans, phi, theta, psi]
                
            elif len(self.go_conversion_rule[i])>4: # on surface
                mol_vec = vec[6*i : 6*i+6]
                surf_idx = int(mol_vec[0]) # remain the same surface index. Output's 1st value.
                face = self.go_conversion_rule[i][ 2+ surf_idx ]
                # Where adsorbate atoms are now:                
                adsorb_location_new = pos_new[ self.go_conversion_rule[i][0] ]
                adsorb_direction_new = pos_new[ self.go_conversion_rule[i][1] ] - adsorb_location_new
                # project adsorb atom on the surface
                v = adsorb_location_new - face[0] # from p0 to adsorbate
                distance = np.dot(v, face[3]) # distance from adsorbent to plane along surf norm. Also 4th value of output
                projected_point = adsorb_location_new - distance*face[3] 
                # Now we first find the intermediate point on p0-p1
                v1 = face[1] - face[0]
                v2 = projected_point - face[2]
                A = np.column_stack([-v1, v2])
                b = face[2] - face[0]
                # Solve A @ [x1, x2] = b using least-squares. Only valid if lines intersect (or nearly do)
                sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                x1, x2 = sol # The 2nd and 3rd value of output
                # Lastly, the rotation angle along surf norm
                mol_new.rotate( adsorb_direction_new, face[3], center=adsorb_location_new)
                pos_new = mol_new.get_positions()
                v_new = pos_new - np.mean(pos_new, axis=0)
                v_old = pos_old - np.mean(pos_old, axis=0)
                for n in range(len(v_new)):
                    v1 = v_old[n] - np.dot(v_old[n], face[3]) * face[3]
                    v2 = v_new[n] - np.dot(v_new[n], face[3]) * face[3]
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        break
                    else:
                        raise ValueError("No atom suitable for computing rotation angle (all aligned with axis?)")
                v1 = v1/ np.linalg.norm(v1) 
                v2 = v2/ np.linalg.norm(v2)
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                cross = np.cross(v1, v2)
                angle_rad = np.arccos(dot)
                sign = np.sign(np.dot(cross, face[3]))
                angle_deg = np.degrees(angle_rad) * sign
                angle_deg = angle_deg % 360 # Normalize to 0-360
                # Final output
                vec_new += [surf_idx, x1, x2, distance, angle_deg, 0]
                
            else:
                raise ValueError(f'go_conversion_rule has a strange length: {len(self.go_conversion_rule[i])}')

        return np.array(vec_new)


    # The obj func for energy computing
    # Assign possible ways to compute energy
    def obj_func_compute_energy(self, vec, computing_id, save_output_directory):
        atoms = self.vector_to_cluster(vec)
        
        if self.save_output_level == 'Simple' and self.calculator_type == 'ase':
            pass
        else:
            # Each specific job folder
            new_cumpute_directory = os.path.join(save_output_directory,computing_id)
            os.makedirs( new_cumpute_directory, exist_ok=True)   
            # Log the starting structure
            write( os.path.join(new_cumpute_directory, 'start.xyz'), atoms, format='xyz' )
        
        # if use coarse calc to pre-relax
        if self.if_coarse_calc:
            atoms.calc = self.coarse_calculator 
            if self.save_output_level == 'Full':
                dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
                dyn = BFGS(atoms, logfile=dyn_log ) 
            elif self.save_output_level == 'Simple':
                dyn = BFGS(atoms)
            else:
                raise ValueError('Saving output level keyword is not supported')
            if self.coarse_calc_constraint is not None:
                atoms.set_constraint( self.coarse_calc_constraint )
            dyn.run( fmax=self.coarse_calc_fmax, steps=self.coarse_calc_step )
            if self.save_output_level == 'Simple' and self.calculator_type == 'ase':
                pass
            else:
                write( os.path.join(new_cumpute_directory, 'coarse_final.xyz'), atoms, format='xyz') 
                
            vec = self.cluster_to_vector( atoms, vec ) # update vec since we optimized the structure   
                          
        # The fine optimization
        start_time = time.time()
        if self.calculator_type == 'ase':   # To use ASE calculator
            atoms.calc = self.calculator
            # If anything happens (e.g. SCF not converged due to bad structure), return a fake high energy
            if self.geo_opt_para is not None:
                if self.save_output_level == 'Full':
                    dyn_log = os.path.join(new_cumpute_directory, 'opt.log') 
                    dyn = BFGS(atoms, logfile=dyn_log ) 
                elif self.save_output_level == 'Simple':
                    dyn = BFGS(atoms)
                else:
                    raise ValueError('Saving output level keyword is not supported')
                ##traj = Trajectory( os.path.join(new_cumpute_directory, 'opt.traj'), 'w', atoms)
                ##dyn.attach(traj.write, interval=10)
                try:
                    fmax = self.geo_opt_para['fmax']
                    steps = self.geo_opt_para['steps']
                except:
                    raise ValueError('Geo Opt cannot be done due to missing parameter fmax or steps' )
                if 'ase_constraint' in self.geo_opt_para:
                    atoms.set_constraint( self.geo_opt_para['ase_constraint'] )
                try:                    
                    dyn.run( fmax=fmax, steps=steps )
                    energy = atoms.get_potential_energy()
                    vec = self.cluster_to_vector( atoms, vec )    # Update vec again after opt
                except:
                    energy = 1e7 # Cannot optimize properly to get energy
                    # And vec is not changed
            else:
                try:
                    energy = atoms.get_potential_energy()
                except:
                    energy = 1e8 # Cannot compute energy

        elif self.calculator_type == 'external': # To use external command
            atoms, energy = self.call_external_calculation(atoms, new_cumpute_directory, self.calculator , self.geo_opt_para)
            vec = self.cluster_to_vector( atoms, vec )  
            
        elif self.calculator_type == 'structural': # For structure generation
            energy = 0.0
            
        else:
            raise ValueError('calculator_type is not supported')

        # Vec, structure and energy are all finalized now. They will be saved in db file.
        if self.save_output_level == 'Full':
            np.savetxt(os.path.join(new_cumpute_directory, 'vec.txt'), vec, delimiter=',')  
            write( os.path.join(new_cumpute_directory, 'final.xyz'), atoms )
            np.savetxt(os.path.join(new_cumpute_directory, 'energy.txt'), [energy], delimiter=',')
        
        current_time = np.round(time.time() - start_time , 3)
        print( 'Time cost track for ',computing_id, ' is (in s): ',current_time, ' with energy: ', np.round(energy,3) )  
        
        return  vec, energy, atoms
    
        
    # Call the external tool to compute energy
    def call_external_calculation(self, atoms, job_directory, calculator_command_lines , geo_opt_para_line ):
        """
        Prepare the input script, run and save output to a file, get energy from the output file
        
        Parameters
        ----------
        atoms : ASE obj
            The atoms generated.
        job_directory : str
            This job's directory path. All computations need to be done here.
        calculator_command_lines : str
            The bash commands needed to perform this computation. 
            It can has multiple lines as long as each line is seprated by ;
        geo_opt_para_line : str
            Provide additional controls.

        Returns
        -------
        atoms: ASE atoms
        energy : float
            The energy of this ASE obj (atoms).

        """
        
        # Know which file to use as initial structure
        start_xyz = 'start.xyz'
        if self.if_coarse_calc:
            start_xyz = 'coarse_final.xyz'
        
        # Go to the job folder and update the input 
        current_directory = os.getcwd()
        os.chdir(job_directory)
        calculator_command_lines = calculator_command_lines.replace('{input_xyz}', start_xyz)

        # Compute
        if geo_opt_para_line['method'] == 'xTB':
            if self.save_output_level == 'Full':
                calculator_command_lines += ' > job.log ' # Write results to job.log
                try:
                    result = subprocess.run(calculator_command_lines, shell=True, check=False, capture_output=True, text=True )
                    energy, energy1 = [],[]
                    with open('job.log','r') as f1:
                        for line in f1.readlines():
                            if "TOTAL ENERGY" in line:
                                energy.append( line.split() )
                            if "total energy " in line:
                                energy1.append( line.split() )
                except: # If the job cannot be performed.
                    energy = None  # In case the structure is really bad and you cannot compute its energy. We still want to continue the code.
            elif self.save_output_level == 'Simple':
                try:
                    result = subprocess.run(calculator_command_lines, shell=True, check=False, capture_output=True, text=True )
                    energy, energy1 = [],[]
                    for line in result.stdout.split('\n'):
                        if "TOTAL ENERGY" in line:
                            energy.append( line.split() ) 
                        if "total energy " in line:
                            energy1.append( line.split() )
                except:
                    energy = None
            else:
                raise ValueError('Output saving level keyword is wrong')
                    
            # Get the energy
            if energy is not None:
                if len(energy)>0:
                    energy = float(energy[-1][3]) # last energy. Value is the 4th item
                elif len(energy1)>0:
                    energy = float(energy1[-1][3]) 
                else:
                    energy = 1e12 # Optimization done but no energy written. This should not happen.
            else:
                energy = 1e8
            
            # Get the final structure
            if os.path.exists( 'xtbopt.xyz' ):
                final_xyz = 'xtbopt.xyz'
            elif os.path.exists( 'xtblast.xyz' ):
                final_xyz = 'xtblast.xyz'
            else:
                final_xyz = f'{start_xyz}'

            if self.save_output_level == 'Full':
                shutil.copyfile( f'{final_xyz}' , 'final.xyz' )
            atoms = read(f'{final_xyz}')
                
        elif geo_opt_para_line['method'] == 'CP2K':
            if 'input' in geo_opt_para_line: # Check if CP2K input is ready
                if os.path.exists( geo_opt_para_line['input'] ): # If we provide absolute path
                    CP2K_input = geo_opt_para_line['input']
                elif os.path.exists( os.path.join(current_directory, geo_opt_para_line['input']) ) :# Check job root path
                    CP2K_input = os.path.join(current_directory, geo_opt_para_line['input'])
                else:
                    raise ValueError('CP2K input is not found from key path')                   
                # Run CP2K
                shutil.copyfile( start_xyz , 'data-CP2K-initial.xyz' )
                calculator_command_lines = calculator_command_lines.replace('{input_script}', CP2K_input)
                try:
                    result = subprocess.run(calculator_command_lines, 
                                            shell=True, check=False, 
                                            capture_output=True, text=True
                                            )
                    # Now get the energy from CP2K
                    with open('job.log','r') as f1:
                        energy = [line.split() for line in f1.readlines() if "Total energy: " in line ]
                    energy = float(energy[-1][2]) # last energy. Value is the 3rd item
                    # Get the final xyz from either initial or optimized xyz, depending on input runtype
                    atoms = get_CP2K_run_info(CP2K_input, start_xyz)
                    
                except subprocess.CalledProcessError as e:
                    print( job_directory , e.stderr)
                    energy = 1e8
                    atoms = read(start_xyz)

            else:
                raise NameError('CP2K input is not provided by input key')
        
        elif geo_opt_para_line['method'] == 'Gaussian':
            if 'input' in geo_opt_para_line:
                if os.path.exists( geo_opt_para_line['input'] ):
                    gaus_input = geo_opt_para_line['input']
                elif os.path.exists( os.path.join(current_directory, geo_opt_para_line['input']) ) :# Check job root path
                    gaus_input = os.path.join(current_directory, geo_opt_para_line['input'])
                else:
                    raise ValueError("Gaussian input needs to be provided by geo_opt_para_line['input']= XXX")    
                                    
                with open(start_xyz, 'r') as f1: # read XYZ coordinates and element symbols
                    lines = f1.readlines()
                    content_xyz = '\n'.join( [ '  '.join(line.split()[:4]) for line in lines[2:] ] )
                    natoms = int( lines[0] )
                    elements = [ line.split()[0] for line in lines[2:] ]
                with open(gaus_input,'r') as f2: # read Gaussian input
                    content_ga = f2.read()
                content_ga = content_ga.replace( "{structure_info}", content_xyz)
                with open('gaussian_input', 'w') as f3:
                    f3.write(content_ga)
                    
                calculator_command_lines = calculator_command_lines.replace('{input_script}', 'gaussian_input')
                try:
                    subprocess.run(calculator_command_lines, shell=True, check=False, capture_output=True, text=True)
                    energy = None
                except:
                    print( f' Gaussian failed. Check detail at {job_directory}. Moving on with a fake high energy.' )
                    energy = 1e8
                    atoms = read(start_xyz)
                
                if energy is None: # calculation done successfully
                    with open('job.log','r') as f4:  # Both energy and structure is here
                        lines = f4.readlines()
                        energy, atoms = [],[]
                        for line_idx, line in enumerate(lines):
                            if "SCF Done" in line:
                                energy.append( line.split()[4] )
                            elif "Coordinates (Angstroms)" in line:
                                atoms.append( [ lines[idx].split()[3:] for idx in range(line_idx+3, line_idx+3+natoms) ] )
                                
                    energy = float( energy[-1] )
                    atoms = Atoms( elements, positions = np.array(atoms[-1],dtype=float) )

            else:
                raise NameError('Gaussian input is not provided by input key geo_opt_para_line')
            
        elif geo_opt_para_line['method'] == 'ORCA':
            if 'input' in geo_opt_para_line: # Check if CP2K input is ready
                if os.path.exists( geo_opt_para_line['input'] ): # If we provide absolute path
                    ORCA_input = geo_opt_para_line['input']
                elif os.path.exists( os.path.join(current_directory, geo_opt_para_line['input']) ) :# Check job root path
                    ORCA_input = os.path.join(current_directory, geo_opt_para_line['input'])
                else:
                    raise ValueError('ORCA input is not found from key path')  
                # Run ORCA
                shutil.copyfile( start_xyz , 'data-ORCA-initial.xyz' )
                shutil.copyfile( ORCA_input , 'input-ORCA' )
                calculator_command_lines = calculator_command_lines.replace('{input_script}', 'input-ORCA')
                try:
                    result = subprocess.run(calculator_command_lines, 
                                            shell=True, check=False, 
                                            capture_output=True, text=True
                                            )
                    # Now get the energy from ORCA
                    with open('job.log','r') as f1:
                        energy = [line.split() for line in f1.readlines() if "FINAL SINGLE POINT ENERGY" in line ]
                    energy = float(energy[-1][-1]) # last energy. Value is the last value
                    # Get the final xyz 
                    atoms = read( 'input-ORCA.xyz' )
                except subprocess.CalledProcessError as e:
                    print( 'ORCA Error: ', job_directory , e.stderr)
                    energy = 1e8
                    atoms = read(start_xyz)
            else:
                raise NameError('ORCA input is not provided by input key')

        elif geo_opt_para_line['method'] == 'User':  # If a general way from user
            subprocess.run(calculator_command_lines, shell=True, check=True, capture_output=True, text=True)
            energy = subprocess.run(geo_opt_para_line['get_energy'], shell=True, check=True, capture_output=True, text=True) 
            structure_file = subprocess.run(geo_opt_para_line['get_structure'], shell=True, check=True, capture_output=True, text=True) 
            atoms = read(structure_file)
        else:
            raise ValueError('External calculation setting has wrong values:', calculator_command_lines, geo_opt_para_line )
            
        # Go back to the main folder
        os.chdir(current_directory)
        
        return atoms, energy 
    

