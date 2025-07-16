import sys, os
from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off
from ase.optimize import BFGS
from ase.io import read, write


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
xyz_path = '../xyz_structures'
substrate = os.path.join(xyz_path, 'batio3-cub-7layer-tio.xyz' )
#substrate = 'gm-from-makos.xyz'

atoms = read( substrate )#sys.argv[1] )
#atoms = atoms[[atom.index for atom in atoms if atom.symbol!='Cu']]

#pbc_box=(20.26028, 20.26028, 32.13928)
pbc_box=(17,17,17)
atoms.set_pbc( (True,True,True) )
atoms.set_cell( pbc_box )
            
ase_calculator = mace_mp(model='small', dispersion=False, default_dtype="float64", device='cuda')
atoms.calc = ase_calculator

#dyn_log = os.path.join(new_cumpute_directory, 'coarse-opt.log')
dyn = BFGS(atoms)#, logfile=dyn_log )
dyn.run( fmax=0.2, steps=50 )
write( 'geoopt.xyz', atoms )

# MD section if we want
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.io.trajectory import Trajectory
from ase import units
from ase.md import MDLogger


MaxwellBoltzmannDistribution(atoms, temperature_K=300)

#dyn = VelocityVerlet(atoms, 5 * units.fs) 
dyn = Langevin(atoms, timestep=5*units.fs, temperature_K=300, friction=0.01/units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False,
           peratom=False, mode="w"), interval=10)
#traj = Trajectory('md-nvt.traj', 'w', atoms)
#dyn.attach(traj.write, interval=50)
dyn.run(200)
write( 'md-nvt.xyz', atoms )

# Room temperature simulation (300K, 0.1 fs time step, atmospheric pressure)
dyn = NPTBerendsen(atoms, timestep=0.2*units.fs, temperature_K=300,
                   taut=100 * units.fs, pressure_au=1.01325 * units.bar,
                   taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False,
           peratom=False, mode="a"), interval=10)
dyn.run(200)
write( 'md-npt.xyz', atoms )

atoms.wrap()
write( 'md-np.gro', atoms )
