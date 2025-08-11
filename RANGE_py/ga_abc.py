# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 08:55:16 2025

@author: d2j
"""

import numpy as np
import os
import time
from RANGE_py.input_output import save_structure_to_db, read_structure_from_db, read_structure_from_directory, print_code_info
from RANGE_py.utility import select_max_diversity


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
                 restart_strategy = 'lowest',
                 
                 apply_algorithm = 'ABC_GA',
                 if_clip_candidate = True,
                 early_stop_parameter = None
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
        self.output_database = output_database
        
        self.restart_from_pool = restart_from_pool
        self.restart_strategy =restart_strategy
        
        self.rng = np.random.default_rng()
        self.global_structure_index = 0
        
        self.apply_algorithm = apply_algorithm 
        self.if_clip_candidate = if_clip_candidate

        self.early_stop_parameter = early_stop_parameter

    # Initial colony from random generation if not restarting
    def _init_colony(self, print_interval):
        if print_interval is not None:
            print_code_info('Header')
        if self.restart_from_pool is not None:   # Read existing database
            if os.path.exists(self.restart_from_pool):
                if os.path.isfile(self.restart_from_pool): # From .db file
                    self.x, self.y, names, previous_pool_size = read_structure_from_db( self.restart_from_pool, self.restart_strategy, self.colony_size )
                elif os.path.isdir(self.restart_from_pool):  # From results directory. This may be slow.
                    self.x, self.y, names, previous_pool_size = read_structure_from_directory( self.restart_from_pool, self.restart_strategy, self.colony_size )
                else:
                    raise ValueError(f'{self.restart_from_pool} cannot be read')
                self.global_structure_index  += previous_pool_size
                self.pool_name = list(names)
                self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) # The initial pool
            else:
                raise ValueError(f'{self.restart_from_pool} does not exist to restart. Either start from scratch or make sure this file exists.')
            if print_interval is not None:
                print(f"Initialization from previous generations in {self.restart_from_pool}")
        else:
            lo, hi = self.bounds.T   # each is a 1D array, shape = D
            self.x = lo + (hi-lo)*np.random.rand( self.colony_size*5, self.bounds_dimension )  # get input X, shape = 6N*D
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
            self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) # The initial pool
            # Narrow down X and Y to colony size
            idx = select_max_diversity(self.x, self.y, self.colony_size)
            self.x, self.y = self.x[idx], self.y[idx]
            if print_interval is not None:
                print("Initialization from random generations by SC bees using",  ' '.join([f"{ix}-->{i}" for i,ix in enumerate(idx)]) )
        self.trial = np.zeros( self.colony_size , int)  # trial counter
        # initialize the track of bees: a dict of lists (bees). Every bee: a list of dict, each dict is a structure name:[x,y]
        #track_bees = { n:[ {self.pool_name[n]:[self.x[n],self.y[n]] } ] for n in range(self.colony_size) }
        # This can also be achieved by post-analysis of the compute names. So comment it out.
        
        # The best X and Y at the begining
        best_idx = np.argmin(self.y) 
        self.best_id = np.copy( self.pool_name[best_idx] )
        self.best_y = np.copy( self.y[best_idx] )
        self.best_x = np.copy( self.x[best_idx] )
        self.best_trial = 0
        
    # Generate new candidate around solution i
    def _neighbor_search(self, i): 
        # ------ Differential evolution method
        #k = np.random.choice([j for j in range(self.colony_size) if j != i])
        #phi = np.random.uniform(-1, 1, self.bounds_dimension)
        #v = np.clip(self.x[i] + phi*(self.x[i]-self.x[k]), self.bounds[:,0], self.bounds[:,1])
        # ------ Trigonometric mutation
        p = (np.amax(self.y) - self.y)/(np.amax(self.y)-np.amin(self.y)+1e-8)
        k1,k2,k3 = np.random.choice([j for j in range(self.colony_size) if j != i], size=3, replace=False) 
        p1,p2,p3 = p[k1],p[k2],p[k3]
        v = (self.x[k1]+self.x[k2]+self.x[k3])/3 
        v = v + (p2-p1)*(self.x[k1]-self.x[k2]) + (p3-p2)*(self.x[k2]-self.x[k3]) + (p1-p3)*(self.x[k3]-self.x[k1])
        if self.if_clip_candidate:
            v = np.clip(v, self.bounds[:,0], self.bounds[:,1])
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
        mutate_sigma =self.mutate_sigma*(1 + 5*self.best_trial/self.global_structure_index)
        if np.random.rand() < self.mutate_rate:
            noise = np.random.randn(self.bounds_dimension) * mutate_sigma * (self.bounds[:,1]-self.bounds[:,0])
            if self.if_clip_candidate:
                child = np.clip(child + noise, self.bounds[:,0], self.bounds[:,1])
            else:
                child = child + noise
        return child
        
    def _ga_step(self, iteration_idx, ga_type):
        sorted_y_index = np.argsort(self.y)
        # Find the worst candidates
        worst_idx = sorted_y_index[-self.ga_parents:]
        # select elite parents from the best candidates
        elite_idx = sorted_y_index[:self.ga_parents] 
        parents = self.x[elite_idx]
        # generate offspring ( number of offspring = number of parents )
        offspring, offspring_compute_id = [], []
        while len(offspring) < self.ga_parents:
            if ga_type>0:
                p1, p2 = parents[np.random.choice(self.ga_parents, 2, replace=False)]
            else:
                p1 = parents[np.random.choice(self.ga_parents)]
                p2 = self.best_x[:]
            child  = self._uniform_crossover(p1, p2)
            child  = self._mutate(child)
            offspring.append(child)
        offspring = np.array(offspring)
        # Replace the worst parents by offspring
        y_off = []
        for i, x_off in enumerate(offspring):
            compute_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{iteration_idx}_ga_{worst_idx[i]}' 
            self.global_structure_index += 1
            x_off, y_ , atoms = self.func( x_off , compute_id , self.output_directory )
            save_structure_to_db(atoms, x_off, y_, compute_id , self.output_database )
            offspring[i] = x_off
            y_off.append( y_ )
            offspring_compute_id.append( compute_id )
            
            self.update_GM(x_off, y_, compute_id)
        y_off = np.array(y_off)
        
        self.x[worst_idx] = offspring
        self.y[worst_idx] = y_off
        self.trial[worst_idx] = 0
        return offspring, y_off, offspring_compute_id
    
    def update_GM(self, new_x, new_y, new_id ):
        if new_y < self.best_y:
            self.best_id = str(new_id)
            self.best_y = float(new_y)
            self.best_x = np.asarray(new_x)
            if (self.best_y - new_y)/abs(self.best_y) > 1E-3:
                self.best_trial = 0
        else:
            self.best_trial += 1
        ##print( '--> ', self.best_trial, self.best_y, self.best_id, new_y)

    def early_stop(self, stop_para):
        terminate_early = False
        if stop_para is not None:  # if not empty
            if 'Max_candidate' in stop_para:
                terminate_early = self.global_structure_index > stop_para['Max_candidate'] 
            elif 'Max_ratio' in stop_para and self.global_structure_index >1000:
                terminate_early = self.best_trial/self.global_structure_index > stop_para['Max_ratio']
            elif 'Max_lifetime' in stop_para:
                terminate_early = self.best_trial > stop_para['Max_lifetime']
            else:
                raise  ValueError(f'Early stop only supports key: Max_candidate, Max_ratio, or Max_lifetime. Current early step is: {stop_para}')
        return terminate_early
        
    def summarize_iteration(self, iteration_count, iteration_time, iteration_generation):
        output_line = f"Iteration: {iteration_count:5d} | best Y so far: {np.round(self.best_y,6):16.6f} | Lifetime: {self.best_trial:5d}"
        output_line += f" | Total X: {self.global_structure_index:9d}"
        output_line += f" | Total time cost(s): {round(iteration_time,3):16.2f} | Cost per X(s): {round(iteration_time/(iteration_generation),3):8.2f}"
        return  output_line
        
    # The main loop 
    def run(self, print_interval=None, if_return_results=False):            
        start_time = time.time()

        self._init_colony(print_interval) 
        previous_pool_size = self.global_structure_index # Starting from this number of generations
        
        lo, hi = self.bounds.T
        current_time = time.time() - start_time
        # Kepp log info as we run 
        if print_interval is not None: 
            with open("log_of_RANGE.log", 'a') as f1:
                f1.write( f"Start iteration based on initial pool of {len(self.y)} solutions from {previous_pool_size} candidates. Current time cost: {round(current_time,3)}\n" )
               
        # Approach 1: ABC+GA
        if self.apply_algorithm == 'ABC_GA':
            ga_type = 1
            for it in range(1, self.max_iteration+1):
                #  employed phase
                em_limit = self.best_trial/self.global_structure_index
                for i in range(self.colony_size):
                    if self.rng.random() < em_limit:
                        new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_em_{i}'
                        self.global_structure_index += 1
                        new_x = self._neighbor_search(i)
                        new_y = self._greedy(i, new_x, new_id )
                    
                        self.update_GM(new_x, new_y, new_id )
                    
                        if if_return_results:
                            self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                            self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                            self.pool_name.append(new_id)
                    
                #  hybrid GA phase
                if it % self.ga_interval == 0:
                    new_x_ga, new_y_ga, new_id = self._ga_step( it, ga_type )
                    ga_type = -1*ga_type
                    
                    if if_return_results:
                        self.pool_x = np.append( self.pool_x, new_x_ga, axis=0 )
                        self.pool_y = np.append( self.pool_y, new_y_ga, axis=0 )
                        self.pool_name += new_id
    
                #  onlooker phase
                # Dynamically determine OL phase numbers
                num_of_OL = 1 + int((1-self.best_trial/self.global_structure_index)*self.colony_size/2)
                # ABC/best/2 strategy: DOI: 10.1016/j.ipl.2011.06.002   
                for k in range(num_of_OL):
                    idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                    new_x = self.x[idxs[0]] + self.x[idxs[1]] - self.x[idxs[2]] - self.x[idxs[3]]
                    new_x = self.best_x + self.rng.random()*new_x 
                    if self.if_clip_candidate:
                        new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])
                    idxs = f"{idxs[0]}_{idxs[1]}_{idxs[2]}_{idxs[3]}"
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_ol_{idxs}'
                    self.global_structure_index += 1 
                    new_x, new_y, atoms = self.func(new_x, new_id , self.output_directory )
                    save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                    
                    self.update_GM(new_x, new_y, new_id )
                    
                    if if_return_results:
                        self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                        self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                        self.pool_name.append(new_id)
    
                #  scout phase
                # Dynamically determine threshold: higher threshold when global is improving fast. Otherwise lower to explore more than exploiate.
                sc_limit = int(self.limit * (0.5 + 2 * (1-self.best_trial/self.global_structure_index)))
                sc_limit = max(5, min(sc_limit, 50))
                for i in range(self.colony_size):
                    if self.trial[i] >= sc_limit:
                        self.x[i] = lo + (hi-lo)*np.random.rand(self.bounds_dimension)
                        new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_sc_{i}'
                        self.global_structure_index += 1
                        self.x[i], self.y[i] , atoms = self.func(self.x[i], new_id , self.output_directory )
                        save_structure_to_db(atoms, self.x[i], self.y[i], new_id, self.output_database )
                        self.trial[i] = 0
                        
                        self.update_GM(self.x[i], self.y[i], new_id )
                        
                        if if_return_results:
                            self.pool_x = np.append( self.pool_x, [self.x[i]], axis=0 )
                            self.pool_y = np.append( self.pool_y, [self.y[i]], axis=0 )
                            self.pool_name.append(new_id)
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: EM_threshold={round(em_limit,2)} OL_num={num_of_OL:3d} SC_limit={sc_limit:3d} for best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')

                if self.early_stop(self.early_stop_parameter):
                    break
                
        # Approach 2: ABC in pyGlobOpt
        if self.apply_algorithm == 'ABC_random':
            #sorted_id = np.argsort(self.y)
            #self.y = self.y[sorted_id]
            #self.x = self.x[sorted_id]
            bee_phase_probability = np.array([1,1,1])
            for it in range(1, self.max_iteration+1):
                bee_phase = np.random.choice(['SC','EM','OL'], p=bee_phase_probability/np.sum(bee_phase_probability)) # pick a bee
                new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_{bee_phase}'
                if bee_phase=='EM':
                    new_x = self._neighbor_search(-1)
                elif bee_phase=='OL':
                    idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                    new_x = self.x[idxs[0]] + self.x[idxs[1]] - self.x[idxs[2]] - self.x[idxs[3]]
                    new_x = self.best_x + self.rng.random()*new_x 
                elif bee_phase=='SC':
                    new_x = lo + (hi-lo)*np.random.rand(self.bounds_dimension)
                    
                if self.if_clip_candidate:
                    new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])
                new_x, new_y, atoms = self.func( new_x , new_id, self.output_directory )
                save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                self.update_GM(new_x, new_y, new_id )
                self.global_structure_index += 1
                # Update X and Y so that Y always contains the lowest values
                if new_y < np.amax(self.y):
                    idx = np.argmax(self.y)
                    self.y[idx] = new_y
                    self.x[idx] = new_x
                
                if if_return_results:
                    self.pool_x = np.append( self.pool_x, [new_x], axis=0 )
                    self.pool_y = np.append( self.pool_y, [new_y], axis=0 )
                    self.pool_name.append(new_id)
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:  
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')
        
                if self.early_stop(self.early_stop_parameter):
                    break

        # Approach 3: native GA
        if self.apply_algorithm == 'GA_native':
            for it in range(1, self.max_iteration+1):
                new_x_ga, new_y_ga, new_id = self._ga_step( it, 1 )
                    
                if if_return_results:
                    self.pool_x = np.append( self.pool_x, new_x_ga, axis=0 )
                    self.pool_y = np.append( self.pool_y, new_y_ga, axis=0 )
                    self.pool_name += new_id
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')
                
                if self.early_stop(self.early_stop_parameter):
                    break

        # Run completed
        if print_interval is not None: 
            print(f"Completed with best Y: {self.best_y} at {self.best_id} that has survived last {self.best_trial} times of {self.global_structure_index} generations")
        
        if if_return_results:
            return self.pool_x, self.pool_y, self.pool_name
