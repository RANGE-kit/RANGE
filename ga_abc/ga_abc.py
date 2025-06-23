# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:55:16 2025

@author: d2j
"""

import numpy as np
import os


class GA_ABC():
    def __init__(self, obj_func, bounds,
                 colony_size = 30,  # number of food source ( = employed bees )
                 limit = 10,       # couter threshold for going to scout )
                 max_iteration = 200, # max iteration steps

                 ga_interval = 10 ,  # how often to perform GA
                 ga_parents = 10 ,  # how many elite parents to mate in each GA step. 
                 mutate_rate = 0.1 ,
                 mutat_sigma = 0.1 ,
                 
                 output_directory = 'results',
                 output_header = 'compute_',
                 
                ):
        """
        obf_func: callable, function to provide target for minimize
        bounds: 2D array, the lower/upper limit of variables, shape = D*2
        """
        self.func = obj_func # f
        self.bounds = bounds
        self.bounds_dimension = len(bounds) # D
        self.colony_size = colony_size # N
        self.limit = limit 
        self.max_iteration = max_iteration
        
        self.ga_interval = ga_interval
        self.ga_parents  = ga_parents 
        self.mutate_rate = mutate_rate
        self.mutate_sigma = mutat_sigma
        
        self.output_header = output_header
        self.output_directory = output_directory
        
        self.rng = np.random.default_rng()

    # Initial colony from random generation
    def _init_colony(self):
        lo, hi = self.bounds.T   # each is a 1D array, shape = D
        self.x = lo + (hi-lo)*np.random.rand( self.colony_size, self.bounds_dimension )  # get input X, shape = N*D
        # Replace: self.y = np.apply_along_axis(self.func, 1, self.x)  to save output file
        os.makedirs(self.output_directory, exist_ok=True)
        self.y, self.pool_name = [], []
        for n, initial_x_guess in enumerate(self.x):
            compute_id = self.output_header + f'0_{n}'
            initial_y = self.func(initial_x_guess, compute_id, self.output_directory)
            self.y.append( initial_y )
            self.pool_name.append( compute_id )
        self.y = np.array( self.y )
        self.trial = np.zeros( self.colony_size , int)  # trial counter
        self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) 
        
    # Generate new candidate around solution i
    def _neighbor_search(self, i): 
        k = np.random.choice([j for j in range(self.colony_size) if j != i])
        phi = np.random.uniform(-1, 1, self.bounds_dimension)
        v = np.clip(self.x[i] + phi*(self.x[i]-self.x[k]), self.bounds[:,0], self.bounds[:,1])
        return v
        
    # Greedy update by calculating cost function
    def _greedy(self, i , cand, compute_id):
        y_cand = self.func( cand , compute_id, self.output_directory )
        if y_cand < self.y[i]:
            self.x[i] ,self.y[i] = cand, y_cand
            self.trial[i] = 0
        else:
            self.trial[i] += 1
        return y_cand
    
    # GA functions
    def _uniform_crossover(self, p1, p2):
        mask = np.random.rand(self.bounds_dimension) < 0.5
        child = np.where(mask, p1, p2)
        return child
        
    def _mutate(self, child):
        if np.random.rand() < self.mutate_rate:
            noise = np.random.randn(self.bounds_dimension) * self.mutate_sigma * (self.bounds[:,1]-self.bounds[:,0])
            child = np.clip(child + noise, self.bounds[:,0], self.bounds[:,1])
        return child    
        
    def _ga_step(self, compute_id):
        # select elite parents from the best candidates
        elite_idx = np.argsort(self.y)[:self.ga_parents] 
        parents = self.x[elite_idx]
        # generate offspring ( number of offspring = number of parents )
        offspring, offspring_compute_id = [], []
        while len(offspring) < self.ga_parents:
            p1, p2 = parents[np.random.choice(self.ga_parents, 2, replace=False)]
            child  = self._uniform_crossover(p1, p2)
            child  = self._mutate(child)
            offspring.append(child)
        offspring = np.array(offspring)
        # Replace the worst parents by offspring
        # y_off = np.apply_along_axis(self.func, 1, offspring)
        y_off = []
        for i, x_off in enumerate(offspring):
            y_ = self.func( x_off , compute_id + str(i), self.output_directory )
            y_off.append( y_ )
            offspring_compute_id.append( compute_id + str(i) )
        y_off = np.array(y_off)
        worst_idx = np.argsort(self.y)[-self.ga_parents:]
        self.x[worst_idx] = offspring
        self.y[worst_idx] = y_off
        self.trial[worst_idx] = 0
        return offspring, y_off, offspring_compute_id
        
    # The main loop 
    def run(self, print_interval=None):
        self._init_colony() 
        best_idx = np.argmin(self.y)
        best_x = np.copy( self.x[best_idx] )
        best_y = np.copy( self.y[best_idx] )

        # Keep all the results. Do we need it? The input x will be passed to calculator. 
        # It will be converted to XYZ before calculation. We can save/keep results there.

        lo, hi = self.bounds.T
        for it in range(1, self.max_iteration+1):
            #  employed phase
            for i in range(self.colony_size):
                new_id = self.output_header + f'{it}_em_{i}'
                new_x = self._neighbor_search(i)
                new_y = self._greedy(i, new_x, new_id )
                self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                self.pool_name.append(new_id)

            #  onlooker phase
            # ABC/best/2 strategy: DOI: 10.1016/j.ipl.2011.06.002             
            for k in range(self.colony_size):
                idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                new_x = self.x[idxs[0]] + self.x[idxs[1]] - self.x[idxs[2]] - self.x[idxs[3]]
                new_x = self.x[best_idx] + self.rng.random() * new_x
                new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])
                new_id = self.output_header + f'{it}_ol_{k}_pick_{best_idx}'
                new_y = self._greedy(best_idx, new_x, new_id )
                self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                self.pool_name.append(new_id)

            #  scout phase
            for i in range(self.colony_size):
                if self.trial[i] >= self.limit:
                    self.x[i] = lo + (hi-lo)*np.random.rand(self.bounds_dimension)
                    new_id = self.output_header + f'{it}_sc_{i}'
                    self.y[i] = self.func(self.x[i], new_id , self.output_directory )
                    self.trial[i] = 0
                    self.pool_x = np.append( self.pool_x, [self.x[i]], axis=0 )
                    self.pool_y = np.append( self.pool_y, [self.y[i]], axis=0 )
                    self.pool_name.append(new_id)

            #  hybrid GA phase
            if it % self.ga_interval == 0:
                print( 'Calling GA at iteration: ', it )
                new_x_ga, new_y_ga, new_id = self._ga_step( self.output_header + f'{it}_ga_' )
                self.pool_x = np.append( self.pool_x, new_x_ga, axis=0 )
                self.pool_y = np.append( self.pool_y, new_y_ga, axis=0 )
                self.pool_name += new_id

            # This is the best value after this iteration
            if self.y.min() < self.y[best_idx]:
                best_idx = self.y.argmin()
                
            # Update all-time minimum to this iteration minimum if needed.
            # This may also be used for early termination in future.
            if self.y[best_idx] < best_y:
                best_x = np.copy( self.x[best_idx] )
                best_y = np.copy( self.y[best_idx] )

            if print_interval is not None: 
                if it == 1 or it % print_interval == 0:
                    output_line = f"Iteration {it:6d} | best iteration y = {self.y[best_idx]:.9g} | best all-time y = {best_y:.9g}"
                    output_line += f" | Total structures considered: {len(self.pool_y.flatten()) } "
                    print(output_line)
        
        return best_x, best_y, self.pool_x, self.pool_y, self.pool_name