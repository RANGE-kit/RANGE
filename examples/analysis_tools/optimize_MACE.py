import os, argparse
from mace.calculators import MACECalculator, mace_anicc, mace_mp, mace_off
from ase.optimize import BFGS
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixedPlane, FixAtoms
import torch
from tqdm import tqdm
from pathlib import Path


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Provide user input
try:
    data_file = 'refined-PES-samples.xyz'
    traj = read( data_file , index=":" )
    output_name = 'reopt/'
except:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs="+")
    args = parser.parse_args()
    
    traj = [ read(f, index='-1') for f in args.input ]
    output_name = "geoopt-"

#atoms = atoms[[at.index for at in atoms if at.position[2]>11]]

#atoms = read( 'C3N4.poscar' )
#pbc_box=(20.26028, 20.26028, 32.13928)
pbc_box=(21.404922 , 24.706836 , 30.27529) 

#atoms = atoms.repeat((3, 2, 1))
#write( 'out.xyz', atoms)
#exit()

c = FixAtoms(indices=[at.index for at in traj[0] if at.symbol in ['N']])
#c = FixedPlane( [at.index for at in atoms if at.position[2]<2 ],  direction=[0, 0, 1] )
#atoms.set_constraint(c)


model_path = '/global/cfs/cdirs/m4621/difan/mace_foundation_models/mace-mh-1.model'
dispersion_D3 = True
#device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Use resource: ', device)
model_head = 'omat_pbe'
ase_calculator = mace_mp(model=model_path, dispersion=dispersion_D3, default_dtype="float64", device=device, head=model_head)

output_traj = []
for n,atoms in tqdm(enumerate(traj), total=len(traj), desc="Analyzing: "):
#for n,atoms in enumerate(traj):
    output_path = f'{output_name}frame-{n}.xyz'
    p = Path(output_path)

    if p.is_file(): # or n<3200:
        print( 'Pass', n )
    else:
        #atoms.set_pbc( (True,True,True) )
        #atoms.set_cell( pbc_box )

        #atoms.set_constraint(c)
        atoms.calc = ase_calculator
        
        dyn = BFGS(atoms, logfile=None)#, logfile='opt.log' or None )
        dyn.run( fmax=0.02, steps=500 )
        
        pe = atoms.get_potential_energy()
        atoms.calc = SinglePointCalculator(atoms, energy=pe)
        
        #output_traj.append( atoms )

        #print( '---->', n, pe )
        write( output_path, atoms )

#write( output_name, output_traj )

exit()

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
