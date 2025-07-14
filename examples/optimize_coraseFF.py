import sys
from RANGE_py.cluster_model import cluster_model
from RANGE_py.energy_calculation import RigidLJQ_calculator
from ase.optimize import BFGS
from ase.io import read, write


atoms = read( sys.argv[1] )

carbon = '/gpfs/wolf2/cades/phy191/scratch/d2j/global_opt/c60-range/C.xyz'
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


pos_old = atoms.get_positions()
atoms.rattle( stdev=0.1 )
pos_new = atoms.get_positions()
write( 'tmp1.xyz', atoms )

print( pos_new - pos_old )

coarse_calc = RigidLJQ_calculator(cluster_template, charge=[0.0]*len(atoms), epsilon=0.1, sigma=3.4, cutoff=3)
atoms.calc = coarse_calc
#dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
dyn = BFGS(atoms)#, logfile=dyn_log )
dyn.run( fmax=1, steps=20 )
write( 'tmp2.xyz', atoms )

#            vec = self.cluster_to_vector( atoms, vec ) # update vec since we optimized the structure
