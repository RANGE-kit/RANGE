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
parser.add_argument('--xyz',   type=str, help='Input XYZ file')
parser.add_argument('--frame', type=int, default='-1' , nargs="+", help='Input XYZ frame ID: a space-separated list of integers. Default: Use log file')
parser.add_argument('--log',   type=str, default='', help='Energy log file associated with XYZ file. Works by default when no frame ID was used.')
parser.add_argument('--separate',   type=str, default='no', help='Save separated files?')
args = parser.parse_args()

if isinstance(args.frame, int):
    args.frame = [args.frame]
print( args )

# Red frames
traj = read( args.xyz, index=":")
output_traj = []

if args.frame[0]<0 and os.path.exists( args.log ): # By default, use log file
    with open(log,'r') as f1:
        lines = f1.readlines()
    frames = [ n for n in range(1,len(lines)) if "~~~" in lines[n] ] # Find group marker and get its line number
    frames_tail = frames[1:] + [len(lines)]
    for start,final in zip(frames,frames_tail):
        group_traj = []
        for line in lines[start:final]:
            group_traj.append( traj[ int(line.split()[0]) ] )
        output_traj.append( group_traj )
elif args.frame[0]>-1:
    frame_id = args.frame
    for i in frame_id:
        output_traj.append( traj[i] )
    output_traj = [output_traj] # To have a consistent format
else:
    raise ValueError("Input setting is not supported... ")

# Write frames
if args.separate == 'no':
    for n, group in enumerate(output_traj):
        write(f'captured_group_{n}.xyz', group)
elif len(output_traj)==1:
    output_traj = output_traj[0]
    for i,atoms in zip(frame_id,output_traj):
        write(f'captured_frames_{i}.xyz', atoms)
else:
    raise ValueError("Input setting is not supported... ")


