#!/bin/bash
#SBATCH -A  m4621
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --ntasks-per-node 128
#SBATCH --constraint cpu
#SBATCH --qos debug
#SBATCH --time 0:30:00

ulimit -s unlimited
export OMP_STACKSIZE=4G
export OMP_NUM_THREADS=64
export OMP_MAX_ACTIVE_LEVELS=1

python   inbox_C60_xtb.py 

