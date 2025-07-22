# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:55:16 2025

@author: d2j
"""

import numpy as np
import os
import time
from RANGE_py.input_output import save_structure_to_db, read_structure_from_db, read_structure_from_directory, clean_directory


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
                 output_database = 'structure_pool.db',
                 
                 restart_from_pool = None,
                 restart_strategy = 'lowest'
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
        if output_database is not None:
            self.output_database = output_database
        
        self.restart_from_pool = restart_from_pool
        self.restart_strategy =restart_strategy
        
        self.rng = np.random.default_rng()
        self.global_structure_index = 0

    # Initial colony from random generation if not restarting
    def _init_colony(self):
        if self.restart_from_pool is not None:   # Read existing database
            if os.path.exists(self.restart_from_pool):
                if os.path.isfile(self.restart_from_pool): # From .db file
                    self.x, self.y, names, previous_pool_size = read_structure_from_db( self.restart_from_pool, self.restart_strategy, self.colony_size )
                elif os.path.isdir(self.restart_from_pool):  # From results directory. This may be slow.
                    self.x, self.y, names, previous_pool_size = read_structure_from_directory( self.restart_from_pool, self.restart_strategy, self.colony_size )
                else:
                    raise ValueError(f'{self.restart_from_pool} cannot be read')
                self.global_structure_index  += previous_pool_size
                self.trial = np.zeros( self.colony_size , int)  # trial counter
                self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) 
                self.pool_name = list(names)
            else:
                raise ValueError(f'{self.restart_from_pool} does not exist to restart. Either start from scratch or make sure this file exists.')
        else:
            lo, hi = self.bounds.T   # each is a 1D array, shape = D
            self.x = lo + (hi-lo)*np.random.rand( self.colony_size, self.bounds_dimension )  # get input X, shape = N*D
            os.makedirs(self.output_directory, exist_ok=True)
            self.y, self.pool_name = [], []
            for n, initial_x_guess in enumerate(self.x):
                compute_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_0_sc_{n}'
                self.global_structure_index += 1
                initial_x_guess, initial_y, atoms = self.func(initial_x_guess, compute_id, self.output_directory)
                save_structure_to_db(atoms, initial_x_guess, initial_y, compute_id, self.output_database )
                self.x[n] = initial_x_guess
                self.y.append( initial_y )
                self.pool_name.append( compute_id )
            self.y = np.array( self.y )
            self.trial = np.zeros( self.colony_size , int)  # trial counter
            self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) # The initial pool
        
    # Generate new candidate around solution i
    def _neighbor_search(self, i): 
        k = np.random.choice([j for j in range(self.colony_size) if j != i])
        phi = np.random.uniform(-1, 1, self.bounds_dimension)
        v = np.clip(self.x[i] + phi*(self.x[i]-self.x[k]), self.bounds[:,0], self.bounds[:,1])
        return v
        
    # Greedy update by calculating cost function
    def _greedy(self, i , cand, compute_id):
        cand, y_cand, atoms = self.func( cand , compute_id, self.output_directory )
        save_structure_to_db(atoms, cand, y_cand, compute_id, self.output_database )
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
        
    def _ga_step(self, iteration_idx):
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
        y_off = []
        for i, x_off in enumerate(offspring):
            compute_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{iteration_idx}_ga_{i}' 
            self.global_structure_index += 1
            x_off, y_ , atoms = self.func( x_off , compute_id , self.output_directory )
            save_structure_to_db(atoms, x_off, y_, compute_id , self.output_database )
            offspring[i] = x_off
            y_off.append( y_ )
            offspring_compute_id.append( compute_id )
        y_off = np.array(y_off)
        worst_idx = np.argsort(self.y)[-self.ga_parents:]
        self.x[worst_idx] = offspring
        self.y[worst_idx] = y_off
        self.trial[worst_idx] = 0
        return offspring, y_off, offspring_compute_id
        
    # The main loop 
    def run(self, print_interval=None):            
        start_time = time.time()

        self._init_colony() 
        best_idx = np.argmin(self.y)
        best_x = np.copy( self.x[best_idx] )
        best_y = np.copy( self.y[best_idx] )

        # Keep all the results. Do we need it? The input x will be passed to calculator. 
        # It will be converted to XYZ before calculation. We can save/keep results there.
        lo, hi = self.bounds.T
        current_time = time.time() - start_time
        # Kepp log info as we run 
        with open("log_of_RANGE.log", 'w') as f1:
            f1.write( f"Start iteration based on initial pool of {len(self.y)} solutions. Current time cost: {round(current_time,3)}\n" )
            
        for it in range(1, self.max_iteration+1):
            #  employed phase
            for i in range(self.colony_size):
                new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_em_{i}'
                self.global_structure_index += 1
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
                new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_ol_{k}_pick_{best_idx}'
                self.global_structure_index += 1
                new_y = self._greedy(best_idx, new_x, new_id )
                self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                self.pool_name.append(new_id)

            #  scout phase
            for i in range(self.colony_size):
                if self.trial[i] >= self.limit:
                    self.x[i] = lo + (hi-lo)*np.random.rand(self.bounds_dimension)
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_sc_{i}'
                    self.global_structure_index += 1
                    self.x[i], self.y[i] , atoms = self.func(self.x[i], new_id , self.output_directory )
                    save_structure_to_db(atoms, self.x[i], self.y[i], new_id, self.output_database )
                    self.trial[i] = 0
                    self.pool_x = np.append( self.pool_x, [self.x[i]], axis=0 )
                    self.pool_y = np.append( self.pool_y, [self.y[i]], axis=0 )
                    self.pool_name.append(new_id)

            #  hybrid GA phase
            if it % self.ga_interval == 0:
                #print( 'Calling GA at iteration: ', it )
                new_x_ga, new_y_ga, new_id = self._ga_step( it )
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
                    current_time = time.time() - start_time
                    output_line = f"Iteration {it:5d} | best Y= {np.round(self.y[best_idx],6):16.6f} | All-time best Y= {np.round(best_y,6):16.6f}"
                    output_line += f" | Total X: {self.global_structure_index:9d}"
                    output_line += f" | Total time cost(s): {round(current_time,3):16.2f} | Cost per X(s): {round(current_time/(self.global_structure_index),3):8.2f}"
                    with open("log_of_RANGE.log", 'a') as f1:
                        f1.write( output_line+'\n' )
        
        return self.pool_x, self.pool_y, self.pool_name
