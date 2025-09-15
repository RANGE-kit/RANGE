#!/bin/bash
#SBATCH -A m3269
#SBATCH --nodes 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --ntasks-per-node 4
#SBATCH --constraint gpu
#SBATCH --qos regular
#SBATCH --time 47:55:00


ulimit -s unlimited
export OMP_STACKSIZE=4G
#export OMP_NUM_THREADS=128
export OMP_MAX_ACTIVE_LEVELS=1
#export OMP_NUM_THREADS=16

python   input-range.py

