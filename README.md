# RANGE

RANGE: a Robust Adaptive Nature-inspired Global Explorer of potential energy surfaces

A python code using hybrid ABC+GA algorithm to explore potential energy surfaces and search low-energy structures for chemistry and materials science. It can use a third-party computational software as the energy evaluator.

Publication: Difan Zhang, Małgorzata Z. Makoś, Roger Rousseau, Vassiliki-Alexandra Glezakou; RANGE: A robust adaptive nature-inspired global explorer of potential energy surfaces. J. Chem. Phys. 21 October 2025; 163 (15): 152501. https://doi.org/10.1063/5.0288910

## Installation

The Atomic Simulation Environment (ASE) package is required for certain structural manipulation and to use the ASE calculator. ASE installation can be found: https://ase-lib.org/install.html

To install RANGE (with `pip`), download the code, then go to the root directory and do: 
```bash
pip install .
```

## Usage

Examples are provided in the "examples" directory. The "input_detailed_example.py" provides additional explanation of keywords. Examples using different calculators are provided in various folders.

To breifly summarize, here are the minimal steps to setup a RANGE search using MACE ASE calculator to build a copper cluster on BaTiO3 surface:

1. Load necessary library.
```bash
from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

from mace.calculators import mace_mp

from ase.constraints import FixAtoms

```

2. Setup the modeling system.
```bash
# Here we consider a BaTiO3 substrate and 12 copper atoms in the modeling system
substrate = 'BaTiO3-cub-7layer-TiO.xyz'
copper = 'Cu.xyz'
input_molecules = [substrate, copper]  
input_num_of_molecules = [1,12]  
# We put BaTiO3 at its current position, and put 12 Cu atoms into a box region
input_constraint_type = ['at_position','in_box']  
input_constraint_value = [(0,0,0,0,0,0),(-2.5,-2.5,7.5, 2.5,2.5,12.5) ]  # Define the BaTiO3 position, and box position
# Then put everything together into a "cluster" class. The modeling system considers PBC.
cluster = cluster_model(input_molecules, input_num_of_molecules, input_constraint_type, input_constraint_value,
                        pbc_box=(20.26028, 20.26028, 32.13928),  # For BaTiO3 slab's PBC
                        )
# Generate modeling system information for further use
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()  
```

3. Setup the calculation method.
```bash
# If needed, we freeze some atoms to accelerate calculations. This is here just to show the option and it not always needed.
# More control on contraints can be found in examples/MACE_calc/input_Nd_solvation.py
ase_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.symbol != 'Cu'])

# We setup a built-in calculator for coarse, pre-optimization to accelerate geometry optimization and avoid bad initial structures.
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0, 
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=ase_constraint)
                            
# For fine optimization, we use MACE model via its ASE interface
model_path = 'mace-mpa-0-medium.model' # or your own path to the model
ase_calculator = mace_mp(model=model_path, dispersion=False, default_dtype="float64", device='cuda')
geo_opt_parameter = dict(fmax=0.05, steps=100, ase_constraint=ase_constraint)
# Put everything together to setup the calculation method
computation = energy_computation(templates = cluster_template, 
                                 go_conversion_rule = cluster_conversion_rule, 
                                 calculator = ase_calculator,
                                 calculator_type = 'ase', 
                                 geo_opt_para = geo_opt_parameter, 
                                 if_coarse_calc = True, 
                                 coarse_calc_para = coarse_opt_parameter,
                                 save_output_level = 'Simple',
                                 )
```

4. Setup the parameters for algorithm, and run it.
```bash
# We initialize the search by 20 bees, with 10 GA bee mutation in every iteration. 
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=20, limit=40, max_iteration=10, initial_population_scaler=2,
                      ga_interval=1, ga_parents=10, mutate_rate=0.5, mutat_sigma=0.05,
                      )
# Now start the search, and print information along the way
optimization.run(print_interval=1)
```

5. Analysis and summary.

    The generated structures are saved into "structure_pool.db" by default. Analysis can be performed using some scripts in examples/analysis_scripts/ or user-customized scripts. Use/revise as needed.
```bash
# analysis_output_energy.py directly analyzes the results (from db file if exsits, otherwise the result folder), generates a summary log and energy profile figure, 
#    and a XYZ trajectory with energy sorted if needed. To use, under the same path of the RANGE search job:
python analysis_output_energy.py  # No args

# The "clean_structure_pool.py" provides a more detailed structure analysis than analysis_output_energy.py
#   It reads the output db file, and furhter narrows down structures based on connectivity and/or structure similarity and/or other conditions. To use:
python  clean_structure_pool.py   # +args (See first section of the script)

# The "capture_snapshots_from_frameID.py" extracts certain frames into a smaller trajectory file for further analysis. 
python  capture_snapshots_from_frameID.py   # +args (See first section of the script)

# For analytical equations, see example at examples/Object_equation/simple_equation.py. 
# The code can search the minima and return the explored points for immediate further analysis by:
all_X_vec, all_Y, all_name = opt.run(print_interval=1, if_return_results=True)

```


## Acknowledgment

This work is supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, Chemical Sciences, Geosciences, and Biosciences Division, Catalysis Science Program, under Grant No. ERKCC96.

