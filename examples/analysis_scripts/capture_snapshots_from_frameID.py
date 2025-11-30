# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:09:47 2025

@author: d2j
"""

import numpy as np
#import matplotlib.pyplot as plt
import os, argparse

from ase.io import read, write
from ase.db import connect


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--xyz',   type=str, default='structure_pool_sorted_clean.xyz', help='Input XYZ file')
parser.add_argument('--frame', type=int, default='0' , nargs="+", help='Input XYZ frames. Space-separated list of integers')
parser.add_argument('--separate',   type=str, default='no', help='Save separated files')
args = parser.parse_args()
print( args )

# Get frames
traj = read( args.xyz, index=":")

log = 'energy_summary_sorted_clean.log'
if args.frame==0 and os.path.exists( log ): # By default, use log file
    with open(log,'r') as f1:
        lines = f1.readlines()
    frames = [ int(lines[n+1].split()[0]) for n in range(len(lines)) if "vvv" in lines[n] ]
    traj = [ traj[f] for f in frames ]
else:
    try:
        frames = args.frame
        traj = [ traj[f] for f in frames ]
    except ValueError:
        print('Cannot be convert input into valid frames')

if args.separate == 'no':
    write('captured_all_frames.xyz', traj)
else:
    for f,t in zip(frames, traj):
        write(f'captured_frames_{f}.xyz', t)


exit()

