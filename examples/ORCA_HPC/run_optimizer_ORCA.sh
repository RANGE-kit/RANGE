#!/bin/bash
#SBATCH -J  RANGE
#SBATCH -A  PHY191
#SBATCH --nodes 1
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH -o output-%J.out
#SBATCH -e error-%J.out

echo  $SLURM_JOBID

##module load gcc
module load aocc/4.1.0
module load openmpi

ulimit -s unlimited
export OMP_STACKSIZE=4G
export OMP_NUM_THREADS=64
export OMP_MAX_ACTIVE_LEVELS=1

python   inbox_water.py 

