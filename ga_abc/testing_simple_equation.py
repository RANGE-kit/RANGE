# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:57:44 2025

@author: d2j
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from ga_abc import GA_ABC

# Test the code with a simple math equation
if __name__ == "__main__":
    
    # Target function
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
        return y
        
    # Plot the function
    xdat = np.linspace(-4, 4, num=100)
    ydat = [ target_function(x) for x in xdat ]
    plt.plot(xdat, ydat)

    # Find minimum point
    dim     = 1
    bounds  = np.array( [ [-4, 4] ]*dim )

    np.random.seed(0)    
    opt = GA_ABC(target_function, bounds, 
                colony_size=5, limit=10, max_iteration=50, 
                ga_interval=10, ga_parents=5, mutate_rate=0.2, mutat_sigma=0.05,
                output_directory = 'results',
                )
    best_x, best_y, all_x, all_y, all_name = opt.run(print_interval=1)

    # Plot the search trajectory
    for dx,dy in zip(all_x , all_y):
        plt.plot( dx[0], dy, color='r', marker='o', ms=2 )
    plt.plot(best_x, best_y, color='g', marker='*', ms=5)
    plt.show()
    
    print("\nBest solution found:")
    print(" x =", np.round(best_x, 4))
    print(" f =", best_y)

    print( all_x.shape, all_y.shape )