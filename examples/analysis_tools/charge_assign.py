# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:50:42 2026

@author: d2j
"""

from ase.io import read, write
import argparse
import numpy as np


""" ------- Useful data ----------"""
data = ("Water:  HOH = [ 0.4, -0.8, 0.4 ]       \n"
        "Nd3+:   2.4 (rescaled from 3*0.8)      \n"
        "NO-:    OOON = -0.4, -0.4, -0.4, 0.4   \n"
        "OH-:    HO = 0.1, -0.9                 \n"
        )

parser = argparse.ArgumentParser( description="Assign atomic charges", formatter_class=argparse.RawTextHelpFormatter )
parser.add_argument('--input',  type=str, help='Input structure file name')
parser.add_argument('--charge', type=float, nargs="+", default=None, help='The atomic charges to be add. Some useful data:\n'+data)
args = parser.parse_args()

# name of structure file
structure_inp = args.input 
atoms = read( structure_inp )

if atoms.has('initial_charges'):
    print('Structure file contains atomic charges already.')
else:
    # ------ Customized charge assignment -----------
    if args.charge is not None:
        atomic_chrages = args.charge 
        print('Use charge from input line')
    else:
        try:
            atomic_chrages = []  # [0.4, -0.8, 0.4]*233
            print('Use charge from script')
        except:
            print("Charge assignment is failed in any way...")
    
    atoms.set_initial_charges(atomic_chrages)
    # ----------------------------------------------
    
    structure_out = 'charged-'+structure_inp
    write(structure_out, atoms) 
    

