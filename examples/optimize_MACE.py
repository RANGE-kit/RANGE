import sys
from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off
from ase.optimize import BFGS
from ase.io import read, write


atoms = read( sys.argv[1] )

ase_calculator = mace_mp(model='small', dispersion=True, default_dtype="float64", device='cuda')
atoms.calc = ase_calculator

#dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
dyn = BFGS(atoms)#, logfile=dyn_log )
dyn.run( fmax=1, steps=20 )
write( 'tmp2.xyz', atoms )

#            vec = self.cluster_to_vector( atoms, vec ) # update vec since we optimized the structure
