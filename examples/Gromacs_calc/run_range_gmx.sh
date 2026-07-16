#!/bin/bash
#SBATCH --image docker:nersc/gromacs:23.2
#SBATCH -C cpu
#SBATCH -t 47:50:00
#SBATCH -J GMX_GO
#SBATCH -A m3269
#SBATCH -N 1
#SBATCH -q regular

#export GMX_ENABLE_DIRECT_GPU_COMM=true
#export OMP_NUM_THREADS=32
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads

python input_RANGE.py

