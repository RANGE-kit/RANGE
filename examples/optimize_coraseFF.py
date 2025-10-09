from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import RigidLJQ_calculator
from ase.optimize import BFGS
from ase.io import read, write
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt


atoms = read( 'compute_000308_round_15_ol_4_pick_0/start.xyz' )

carbon = 'xyz_structures/C.xyz'
input_molecules = [carbon]
input_num_of_molecules = [60]
input_constraint_type = ['in_sphere_shell']
input_constraint_value = [ (0,0,0, 4,4,4, 0.25) ]
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        #pbc_box=(22.90076, 23.00272, 31.95000),
                        )
cluster.init_molelcules()
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

coarse_calc = RigidLJQ_calculator(cluster_template, charge=0, epsilon='UFF', sigma='UFF', cutoff=10)
atoms.calc = coarse_calc

#dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
dyn = BFGS(atoms, maxstep=0.3 )#, logfile=dyn_log )
dyn.run( fmax=10, steps=20 )

dists = atoms.get_all_distances()
np.fill_diagonal(dists, 999)
print( np.amin(dists) )

write( 'tmp2.xyz', atoms )