#!/bin/bash
#SBATCH --image docker:cp2k/cp2k:2022.1
#SBATCH -A m3269
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2 
#SBATCH --ntasks-per-node 128 
#SBATCH --constraint cpu 
#SBATCH --qos debug
#SBATCH --time 0:30:00 

python  testing_molecular_modeling.py
