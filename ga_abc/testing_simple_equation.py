# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:57:44 2025

@author: d2j
"""

import numpy as np
import matplotlib.pyplot as plt

from ga_abc import GA_ABC

# Test the code with a simple math equation
if __name__ == "__main__":
    
    # Target function
    def target_function(x): 
        #return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)   # rosenbrock
        return np.sum(x**6 -15*x**4 + x**3 + 32*x**2 + 20 )  # Use np.sum to get around the 1-D case
        
    # Plot the function
    xdat = np.linspace(-4, 4, num=100)
    ydat = [ target_function(x) for x in xdat ]
    plt.plot(xdat, ydat)

    # Find minimum point
    dim     = 1
    bounds  = np.array( [ [-4, 4] ]*dim )

    np.random.seed(0)
    opt = GA_ABC(target_function, bounds, 
                colony_size=5, limit=2000, max_iteration=20, 
                ga_interval=5, ga_parents=5, mutate_rate=0.2, mutat_sigma=0.05)
    best_x, best_y, all_x, all_y = opt.run(verbose=True)

    # Plot the search trajectory
    for dx,dy in zip(all_x , all_y):
        plt.plot( dx[0], dy, color='r', marker='o', ms=2 )
    plt.show
    
    print("\nBest solution found:")
    print(" x =", np.round(best_x, 4))
    print(" f =", best_y)

    print( all_x.shape, all_y.shape )