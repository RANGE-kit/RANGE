# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:57:44 2025

@author: d2j
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from RANGE_py.ga_abc import GA_ABC

"""
# One-dimension Target function
def target_function(x, computing_id=None, save_output_directory=None): 
    # Use np.sum to get around the 1-D case 
    y = np.sum(x**6 -15*x**4 + x**3 + 32*x**2 + 20 )  
    #return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)   # rosenbrock
    # Each specific job folder to write
    if False:
        new_cumpute_directory = os.path.join(save_output_directory,computing_id)
        os.makedirs( new_cumpute_directory, exist_ok=True) 
        with open( os.path.join(new_cumpute_directory, 'y.log'), 'w' ) as f1_out:
            f1_out.write( f'{x} {y} \n' )
    return x, y, None
    
# See the space
xdat = np.linspace(-4, 4, num=100)
ydat = [ target_function(x)[1] for x in xdat ]
plt.plot(xdat, ydat)

# Search minima
dim     = 1
bounds  = np.array( [ [-4, 4] ]*dim )
    
opt = GA_ABC(target_function, bounds, 
             colony_size=5, limit=10, max_iteration=30, 
             ga_interval=10, ga_parents=5, mutate_rate=0.2, mutat_sigma=0.05,
             output_directory = 'results',
             )
all_x, all_y, all_name = opt.run(print_interval=10)

# Plot explored points
for dx,dy in zip(all_x , all_y):
    plt.plot( dx[0], dy, color='r', marker='o', ms=2 )

plt.show()
"""

# Two-dimensional Müller-Brown PES 
def target_function( input_vec, computing_id=None, save_output_directory=None ):
    x,y= input_vec
    # Parameters for the Müller-Brown potential
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    """
    | Minimum | (x, y)          | Energy V(x,y) |
    | ------- | --------------- | ------------- |
    | **A**   | (−0.558, 1.442) | −146.700      |
    | **B**   | ( 0.623, 0.028) | −108.167      |
    | **C**   | (−0.050, 0.467) | −80.768       |
    """

    V = 0
    for Ai, ai, bi, ci, xi, yi in zip(A, a, b, c, x0, y0):
        V += Ai * np.exp(ai * (x - xi)**2 + bi * (x - xi) * (y - yi) + ci * (y - yi)**2)
    return input_vec, V, None


# See the space
xdat = np.linspace(-1.5, 1., 500)
ydat = np.linspace(-0.5, 2.0, 500)
xdat, ydat = np.meshgrid(xdat, ydat)
value = np.array([ target_function( [x,y] )[1] for x,y in zip(xdat.flatten(), ydat.flatten()) ])
value = np.reshape(value, xdat.shape)

fig, axs = plt.subplots(1,2,figsize=(8,3),tight_layout=True)
axs[0].contourf(xdat, ydat, value, levels=100, cmap='terrain')

# Mark GM
x, y = [-0.558, 1.442]
gm = target_function( [x,y] )[1]
axs[0].plot( x,y, color='m', marker='*', ms=10 )

# Search minima
bounds  = np.array([(-1.5, 1), (-0.5, 2)])
opt = GA_ABC(target_function, bounds, 
             colony_size=5, limit=10, max_iteration=50, 
             ga_interval=1e10, ga_parents=5, mutate_rate=0.2, mutat_sigma=0.05,
             output_directory = 'results',
             )
all_vec, all_V, all_name = opt.run(print_interval=None)
# Plot explored points
for x in all_vec:
    axs[0].plot( x[0], x[1], color='orange', marker='o', ms=1 )
    
# Minimum at step
V_lo = [ all_V[0] ]
for v in all_V[1:]:
    V_lo.append( np.amin( [v, V_lo[-1]] ) )
xdata = np.arange(len(V_lo))

axs[1].plot( xdata,V_lo, color='gray', lw=1, marker='', ms=10, alpha=0.6 )
axs[1].scatter(xdata,V_lo, fc='none', ec='orange', marker='o', s=5, label='Points', alpha=1)

print(V_lo[-1])

plt.show()
