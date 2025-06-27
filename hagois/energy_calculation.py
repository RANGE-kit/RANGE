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

import subprocess


class energy_computation:
    """
    This contains the functionalities of computing the energy of a molecular structure
    It needs the template of the cluster (generated from generate_bounds), just to know what we are modeling.
    """
    def __init__(self, templates, go_conversion_rule, 
                 calculator, calculator_type, geo_opt_para,
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
                m.euler_rotate(center=center_of_geometry, phi=euler[0], theta=euler[1], psi=euler[2])
                m.translate( t )
                
            # The replacement function to replace atoms. If the atom symbol is X, it means a vaccancy.
            elif len(self.go_conversion_rule[i])==2: 
                t = vec[6*i : 6*i+6]
                substrate_mol = placed.copy() #[ self.go_conversion_rule[i][0] ] # ase obj: mol to be changed
                substrate_atom_id = self.go_conversion_rule[i][1][ round(t[0]) ] # int: atom id to be changed
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
                t = vec[6*i : 6*i+6] 
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

    
    # The obj func for energy computing
    # Assign possible ways to compute energy
    def obj_func_compute_energy(self, vec, computing_id, save_output_directory):
        atoms = self.vector_to_cluster(vec)
        
        # Each specific job folder
        new_cumpute_directory = os.path.join(save_output_directory,computing_id)
        os.makedirs( new_cumpute_directory, exist_ok=True)   
        
        # To use ASE calculator
        if self.calculator_type == 'ase': 
            write( os.path.join(new_cumpute_directory, 'start.xyz'), atoms )
            atoms.calc = self.calculator
            # If anything happens (e.g. SCF not converged due to bad structure), return a fake high energy
            if self.geo_opt_para is not None:
                try:
                    dyn_log = os.path.join(new_cumpute_directory, 'opt.log') 
                    dyn = BFGS(atoms, logfile=dyn_log ) 
                    #traj = Trajectory( os.path.join(new_cumpute_directory, 'opt.traj'), 'w', atoms)
                    #dyn.attach(traj.write, interval=10)
                    dyn.run( fmax=self.geo_opt_para['fmax'], steps=self.geo_opt_para['steps'] )
                    energy = atoms.get_potential_energy()
                except:
                    energy = 1e6
            else:
                try:
                    energy = atoms.get_potential_energy()
                except:
                    energy = 1e7
            write( os.path.join(new_cumpute_directory, 'final.xyz'), atoms )
            np.savetxt(os.path.join(new_cumpute_directory, 'vec.txt'), vec, delimiter=',')
                
        elif self.calculator_type == 'external': # To use external command
            energy = self.call_external_calculation(atoms, new_cumpute_directory, self.calculator , self.geo_opt_para)
            
        elif self.calculator_type == 'structural': # For structure generation
            energy = 0.0
            
        return energy
    
        
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

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        energy : float
            The energy of this ASE obj (atoms).

        """
        
        # Go to the job folder and create the input xyz
        current_directory = os.getcwd()
        os.chdir(job_directory)
        write( 'start.xyz', atoms )
                
        calculator_command_lines = calculator_command_lines.replace('{input_xyz}', 'start.xyz')

        # Compute
        if geo_opt_para_line['method'] in [ 'xtb-gfn2', 'xtb-gfnff' ]:
            if geo_opt_para_line['run_type'] == 'geo_opt':
                calculator_command_lines += ' --opt '
            elif not geo_opt_para_line['run_type'] == 'single_point':
                raise ValueError('Run Type is set to a wrong value')
            calculator_command_lines += ' > job.log ' # Write results to job.log
            os.system( calculator_command_lines )
            try:
                # Now get the energy
                with open('job.log','r') as f1:
                    energy = [line.split() for line in f1.readlines() if "TOTAL ENERGY" in line ]
                energy = float(energy[-1][3]) # last energy. Value is the 4th item
            except:
                energy = 1e8  # In case the structure is really bad and you cannot get energy 
        elif geo_opt_para_line['method'] == 'CP2K':
            if 'input' in geo_opt_para_line: # Check if CP2K input is ready
                if os.path.exists( geo_opt_para_line['input'] ): # If we provide absolute path
                    CP2K_input = geo_opt_para_line['input']
                elif os.path.exists( os.path.join(current_directory, geo_opt_para_line['input']) ) :# Check job root
                    CP2K_input = os.path.join(current_directory, geo_opt_para_line['input'])
                else:
                    raise ValueError('CP2K input is not ready')
                # Run CP2K
                calculator_command_lines = calculator_command_lines.replace('{input_script}', CP2K_input)
                try:
                    result = subprocess.run(calculator_command_lines, 
                                            shell=True, check=True, 
                                            capture_output=True, text=True
                                            )
                    # Now get the energy from CP2K
                    with open('job.log','r') as f1:
                        energy = [line.split() for line in f1.readlines() if "Total energy: " in line ]
                    energy = float(energy[-1][2]) # last energy. Value is the 3rd item
                except subprocess.CalledProcessError as e:
                    print( job_directory , e.stderr)
                    energy = 1e7

            else:
                energy = 1e8 # Not CP2K simulation performed. Just a structure generation.
        else:
            raise ValueError('External calculation setting has wrong values:', geo_opt_para_line )
            
        # Go back to the main folder
        os.chdir(current_directory)
        
        return energy 
    
