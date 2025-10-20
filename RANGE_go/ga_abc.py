# -*- coding: utf-8 -*-
"""
Created on Wed Jun 4 08:55:16 2025

@author: d2j
"""

import numpy as np
import os
import time
from RANGE_go.input_output import save_structure_to_db, read_structure_from_db, read_structure_from_directory, print_code_info
from RANGE_go.utility import select_max_diversity, compute_differences


class GA_ABC():
    def __init__(self, obj_func, bounds,
                 colony_size = 20,  # number of food source ( = employed bees ) 
                 limit = 50,       # couter threshold for going to scout 
                 max_iteration = 100, # max iteration steps 
                 initial_population_scaler = 5, # How many initial population

                 ga_interval = 1 ,  # how often to perform GA
                 ga_parents = 10 ,  # how many GA performed in every activation. 
                 mutate_rate = 0.5 ,
                 mutat_sigma = 0.01 ,
                 
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
        self.initial_population_scaler = initial_population_scaler
        
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
            if isinstance(self.restart_from_pool, str):
                assert os.path.exists(self.restart_from_pool), '{self.restart_from_pool} does not exist!'
                if os.path.isfile(self.restart_from_pool): # From .db file
                    self.x, self.y, names, self.previous_pool_size = read_structure_from_db( self.restart_from_pool, self.restart_strategy, self.colony_size )
                elif os.path.isdir(self.restart_from_pool):  # From results directory. This may be slow.
                    self.x, self.y, names, self.previous_pool_size = read_structure_from_directory( self.restart_from_pool, self.restart_strategy, self.colony_size )
                else:
                    ValueError(f'{self.restart_from_pool} cannot be read')
            elif isinstance(self.restart_from_pool, dict):
                assert len(self.restart_from_pool) == self.colony_size
                self.previous_pool_size = len(self.restart_from_pool)
                self.x, self.y, names = [],[],[]
                for k,v in self.restart_from_pool.items():
                    self.x.append( v[0] )
                    self.y.append( v[1] )
                    names.append( k )
                self.x = np.array( self.x ) 
                self.y = np.array( self.y ) 
            else:
                raise ValueError(f'{self.restart_from_pool} does not exist to restart. Either start from dict, scratch, or a file.')
            self.global_structure_index  += self.previous_pool_size 
            self.pool_name = list(names)
            self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) # The initial pool
            self.atoms_pool = [ None for i in range(len(self.pool_name)) ]
            if print_interval is not None: 
                print(f"Initialization from previous generations in {self.restart_from_pool}")
        else: 
            self.previous_pool_size = 0
            lo, hi = self.bounds.T   # Each is a 1D array, with shape = D 
            self.x = lo + (hi-lo)*np.random.rand( self.colony_size*self.initial_population_scaler, self.bounds_dimension )  # get input X, shape = 6N*D
            os.makedirs(self.output_directory, exist_ok=True)
            self.y, self.pool_name = [], []
            self.atoms_pool = []
            for n, initial_x_guess in enumerate(self.x):
                self.global_structure_index += 1
                compute_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_0_sc_{n}'
                initial_x_guess, initial_y, atoms = self.func(initial_x_guess, compute_id, self.output_directory)
                save_structure_to_db(atoms, initial_x_guess, initial_y, compute_id, self.output_database )
                self.x[n] = initial_x_guess 
                self.y.append( initial_y ) 
                self.pool_name.append( compute_id ) 
                self.atoms_pool.append(atoms)
            self.y = np.array( self.y ) 
            self.pool_x, self.pool_y = np.copy(self.x), np.copy(self.y) # The initial pool 
            # Narrow down X and Y to colony size 
            idx = select_max_diversity(self.x, self.y, self.colony_size) 
            self.x, self.y = self.x[idx], self.y[idx] 
            if print_interval is not None: 
                print("Initialization from random generations by SC bees using",  ' '.join([f"{ix}-->{i}" for i,ix in enumerate(idx)]) ) 
        self.trial = np.zeros( self.colony_size , int)  # trial counter... 
        # The best X and Y at the begining
        best_idx = np.argmin(self.y) 
        self.best_id = np.copy( self.pool_name[best_idx] ) 
        self.best_y = np.copy( self.y[best_idx] ) 
        self.best_x = np.copy( self.x[best_idx] ) 
        self.best_trial = 0 
        
    # Employed bee: Generate new candidate around solution i or trigonometric
    def _neighbor_search(self, i): 
        # ------ Differential evolution 
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
        v_id = f'{k1}_{k2}_{k3}'
        return v, v_id

    # Greedy update for bee locations
    def update_bee_location(self, new_x, new_y, new_id ):
        # If the new X is the same as the current bee, do not duplicate
        diff = compute_differences( self.x, new_x )
        if np.all(diff>1E-3): # All are different
            if new_y > np.amax(self.y): # A bad candidate
                self.trial += 1
                self.best_trial += 1
            else:
                idx = np.argmax(self.y)
                assert len(self.y)==len(self.trial), f'{len(self.y)} {len(self.trial)}'
                self.trial[ self.y < new_y ] += 1
                self.x[idx], self.y[idx], self.trial[idx] = new_x, new_y, 0
                if (self.best_y - new_y)/(abs(self.best_y)+1e-10) > 1E-4: # If a new best Y
                    self.best_trial = 0
                    self.best_x, self.best_y, self.best_id = np.asarray(new_x), float(new_y), str(new_id)
                else:
                    self.best_trial += 1
        else: # No update on current bees due to duplicated new X
            self.best_trial += 1        
    
    # GA functions
    def _greedy_update_GA(self, current_i , new_x, new_y):
        diff = compute_differences( self.x, new_x )
        if np.all(diff>1E-3): # All are different
            if new_y < self.y[current_i]:
                self.x[current_i] ,self.y[current_i] = new_x, new_y
                self.trial[current_i] = 0
            else:
                self.trial[current_i] += 1  
        self.best_trial += 1    # We don't compare to GM here    
            
    def _uniform_crossover(self, p1, p2):
        mask = np.random.rand(self.bounds_dimension) < 0.5
        child = np.where(mask, p1, p2)
        return child
        
    def _mutate(self, child):
        mutate_sigma =self.mutate_sigma*(1 + 4*self.best_trial/self.global_structure_index)
        if np.random.rand() < self.mutate_rate:
            noise = np.random.randn(self.bounds_dimension) * mutate_sigma * (self.bounds[:,1]-self.bounds[:,0])
            if self.if_clip_candidate:
                child = np.clip(child + noise, self.bounds[:,0], self.bounds[:,1])
            else:
                child = child + noise
        return child
    
    def _ga_production(self, ga_type):
        if ga_type>0:
            p = (np.amax(self.y) - self.y)/(np.amax(self.y)-np.amin(self.y)+1e-8)+1e-8
            p1, p2 = np.random.choice(self.colony_size, 2, replace=False, p=p/np.sum(p))
            compute_id = f'_from_{p1}_{p2}'
            p1, p2 = self.x[p1], self.x[p2]
        else:
            p1 = np.random.choice(self.colony_size)
            compute_id = f'_from_{p1}_GM'
            p1, p2 = self.x[p1], self.best_x[:]
        # From two parents to a child
        child  = self._uniform_crossover(p1, p2)
        child  = self._mutate(child)
        return child, compute_id
        
    def _ga_step(self, iteration_idx, ga_type):
        sorted_y_index = np.argsort(self.y)
        # Find the candidates to compete
        #worse_idx = abs(int(self.colony_size/2) - int(self.ga_parents/2)) 
        worse_idx = self.colony_size-1
        
        #elite_idx = sorted_y_index[:int(self.colony_size/2)]  # select elite parents from the best candidates        
        offspring, offspring_compute_id, y_off = [], [], []
        num_of_GA = max( 1, int(self.get_best_ratio()*self.ga_parents) )
        for ii in range(num_of_GA):
            self.global_structure_index += 1 
            new_x, new_id = self._ga_production(ga_type) 
            new_id = self.output_header + f"{self.global_structure_index:06d}_round_{iteration_idx}_ga_{ii}" + new_id 
            new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory)
            save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database ) 
            #self.update_bee_location(new_x, new_y, new_id ) 
            self._greedy_update_GA(sorted_y_index[worse_idx-ii], new_x, new_y)
            
            offspring.append(new_x)
            y_off.append(new_y)
            offspring_compute_id.append(new_id)
        return  np.asarray(offspring), np.asarray(y_off), offspring_compute_id     
                
    def calc_new_candiate(self, candi_x, candi_id, candi_dir):
        if self.if_return_results:
            diff = np.mean( np.abs(self.pool_x - np.array(candi_x))/(np.abs(candi_x)+1e-16) , axis=1 )
            overlap_id = np.where( diff < 1e-8 )[0] # threoshold             
            if len(overlap_id)==0:
                candi_x, candi_y, candi_atoms = self.func(candi_x, candi_id , candi_dir)
                #self.add_to_pool([candi_x], [candi_y], [candi_id], [candi_atoms])  # Only add unique candidates to pool
            else:
                overlap_id = overlap_id[0]
                candi_y, candi_atoms = self.pool_y[overlap_id], self.atoms_pool[overlap_id]
            self.add_to_pool([candi_x], [candi_y], [candi_id], [candi_atoms])  # Or, add all candidates to pool
        else:
            candi_x, candi_y, candi_atoms = self.func(candi_x, candi_id , candi_dir)
        return candi_x, candi_y, candi_atoms

        
    def add_to_pool(self, new_x_to_add, new_y_to_add, new_name_to_add, atoms_to_add):
        self.pool_x = np.append( self.pool_x, new_x_to_add, axis=0 )
        self.pool_y = np.append( self.pool_y, new_y_to_add, axis=0 )
        self.pool_name += new_name_to_add
        self.atoms_pool += atoms_to_add

    def get_best_ratio(self):
        ratio = self.best_trial/(self.global_structure_index - self.previous_pool_size + 1e-10)
        return  ratio
    
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
        self.if_return_results = if_return_results
        start_time = time.time()

        self._init_colony(print_interval) 
        
        lo, hi = self.bounds.T
        current_time = time.time() - start_time
        # Kepp log info as we run 
        if print_interval is not None: 
            with open("log_of_RANGE.log", 'a') as f1:
                f1.write( f"Start iteration based on initial pool of {len(self.y)} solutions from {self.previous_pool_size} candidates. Current time cost: {round(current_time,3)}\n" )

        # Approach 1: native ABC
        if self.apply_algorithm == 'ABC_native':
            for it in range(1, self.max_iteration+1):
                for i in range(self.colony_size): 
                    self.global_structure_index += 1
                    new_x, new_id = self._neighbor_search(-1)
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_em_{i}_from_{new_id}'
                    new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory)   
                    save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                    # Update X and Y so that Y always contains the lowest values
                    self.update_bee_location(new_x, new_y, new_id )
                for i in range(self.colony_size):
                    idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                    new_x = self.x[idxs[0]] + self.x[idxs[1]] - self.x[idxs[2]] - self.x[idxs[3]]
                    new_x = self.best_x + self.rng.random()*new_x 
                    if self.if_clip_candidate:
                        new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])  
                    self.global_structure_index += 1 
                    idxs = f"{idxs[0]}_{idxs[1]}_{idxs[2]}_{idxs[3]}"
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_ol_{i}_from_{idxs}'
                    new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory)
                    save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                    # Update X and Y
                    self.update_bee_location(new_x, new_y, new_id )
                for i in range(self.colony_size):
                    if self.trial[i] >= self.limit:
                        self.global_structure_index += 1
                        new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_sc_{i}'
                        self.x[i] = lo + (hi-lo)*np.random.rand(self.bounds_dimension)                       
                        self.x[i], self.y[i] , atoms = self.calc_new_candiate( self.x[i], new_id , self.output_directory)
                        save_structure_to_db(atoms, self.x[i], self.y[i], new_id, self.output_database )
                        self.trial[i] = 0
                        # Update X and Y
                        self.update_bee_location(self.x[i], self.y[i], new_id )
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:  
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - self.previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')
        
                if self.early_stop(self.early_stop_parameter):
                    break
                
        # Approach 2: ABC in pyGlobOpt/NWPESSE
        elif self.apply_algorithm == 'ABC_random':
            bee_phase_probability = np.array([1,1,1])
            for it in range(1, self.max_iteration+1):
                bee_phase = np.random.choice(['SC','EM','OL'], p=bee_phase_probability/np.sum(bee_phase_probability)) # pick a bee
                self.global_structure_index += 1
                new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_{bee_phase}'
                if bee_phase=='EM':
                    new_x, _ = self._neighbor_search(-1)
                elif bee_phase=='OL':
                    idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                    new_x = self.x[idxs[0]] + self.x[idxs[1]] - self.x[idxs[2]] - self.x[idxs[3]]
                    new_x = self.best_x + self.rng.random()*new_x 
                elif bee_phase=='SC':
                    new_x = lo + (hi-lo)*np.random.rand(self.bounds_dimension)
                    
                if self.if_clip_candidate:
                    new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])
                new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory )
                save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                # Update X and Y so that Y always contains the lowest values
                self.update_bee_location(new_x, new_y, new_id )
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:  
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - self.previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')
        
                if self.early_stop(self.early_stop_parameter):
                    break

        # Approach 3: native GA
        elif self.apply_algorithm == 'GA_native':
            for it in range(1, self.max_iteration+1):
                # Selection one of them, or use both
                new_xs, new_ys, new_ids = self._ga_step( it, 1 )
                new_xs, new_ys, new_ids = self._ga_step( it, -1 ) 
   
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - self.previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print(f'Dynamic info at Iteration {it:5d}: best_Y={self.best_y:16.6} Life={self.best_trial:5d} Gen_size={self.global_structure_index:5d} Ratio={np.round(self.best_trial/self.global_structure_index,2)}')
                
                if self.early_stop(self.early_stop_parameter):
                    break
                
        # Approach 4: Hybrid ABC + GA 
        elif self.apply_algorithm == 'ABC_GA':
            for it in range(1, self.max_iteration+1):
                #print( 'it: ', it )
                # Dynamic employed phase with GA. 
                num_of_EM = int( np.amax( (1, self.get_best_ratio()*self.colony_size/2 ) ) )
                for i in range(num_of_EM):
                    self.global_structure_index += 1
                    new_x, new_id = self._neighbor_search(-1)
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_em_{i}_from_{new_id}'
                    new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory )    
                    save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                    self.update_bee_location(new_x, new_y, new_id )
                        
                # GA step
                if it % self.ga_interval == 0:
                    new_xs, new_ys, new_ids = self._ga_step( it, 1 )
                        
                # Dynamic onlooker phase with GA. 
                # ABC/best/2 strategy: DOI: 10.1016/j.ipl.2011.06.002 
                num_of_OL = int( np.amax( (1, (1-self.get_best_ratio())*self.colony_size/2 ) ) )
                for i in range(num_of_OL): 
                    idxs = np.random.choice(self.colony_size, size=4, replace=False) 
                    new_x = self.best_x + self.rng.random()*(self.x[idxs[0]]+self.x[idxs[1]]-self.x[idxs[2]]-self.x[idxs[3]]) 
                    if self.if_clip_candidate:
                        new_x = np.clip(new_x, self.bounds[:,0], self.bounds[:,1])  
                    self.global_structure_index += 1 
                    idxs = f"{idxs[0]}_{idxs[1]}_{idxs[2]}_{idxs[3]}"
                    new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_ol_{i}_from_{idxs}'
                    new_x, new_y, atoms = self.calc_new_candiate( new_x, new_id , self.output_directory )   
                    save_structure_to_db(atoms, new_x, new_y, new_id, self.output_database )
                    self.update_bee_location(new_x, new_y, new_id )
                            
                # GA step
                if it % self.ga_interval == 0:
                    new_xs, new_ys, new_ids = self._ga_step( it, -1 )

                # Dynamic scout phase with GA
                # Higher threshold when global is improving fast. Otherwise lower to explore more than exploiate.
                sc_limit = round( (1-self.get_best_ratio())*self.limit + self.get_best_ratio()*20 )
                for i in range(self.colony_size):
                    if self.trial[i] >= sc_limit:  # Need to kick this bee
                        self.global_structure_index += 1
                        new_id = self.output_header + f"{self.global_structure_index:06d}" + f'_round_{it}_sc_{i}'
                        # Instead of random, we go to the oppo direction of GA crossover
                        new_x = (self.rng.random()*0.5+0.5)*(lo+hi - self._uniform_crossover(self.best_x[:], self.x[i]) )
                        diff = compute_differences( self.x, new_x )
                        if np.all(diff>1E-2): # All different from current X
                            new_id += '_from_GM'
                        else:
                            new_x = lo + (hi-lo)*np.random.rand(self.bounds_dimension)  # Random
                        self.x[i], self.y[i] , atoms = self.calc_new_candiate( new_x, new_id , self.output_directory )   
                        save_structure_to_db(atoms, self.x[i], self.y[i], new_id, self.output_database )
                        self.trial[i] = 0
                        #self.update_bee_location(self.x[i], self.y[i], new_id )
                        self.best_trial += 1 
                    
                if print_interval is not None: 
                    if it == 1 or it % print_interval == 0:
                        output_line = self.summarize_iteration( it, time.time() - start_time, self.global_structure_index - self.previous_pool_size)
                        with open("log_of_RANGE.log", 'a') as f1:
                            f1.write( output_line+'\n' )
                        print( f'Dynamic info at End of Iteration {it:5d}: EM_num={round(num_of_EM,2)} OL_num={num_of_OL:3d} SC_limit={sc_limit:3d} performed to find best_Y={self.best_y:16.6} Life={self.best_trial:5d} Total_size={self.global_structure_index:5d} Generate_size={self.global_structure_index-self.previous_pool_size:5d} with current Ratio={np.round(self.get_best_ratio(),2)}')
                        #print( f'Bee values : {it:5d}', ' '.join(list(self.y.astype('str'))) )
                if self.early_stop(self.early_stop_parameter):
                    break
                
        # Approach unknown
        else:
            raise ValueError('apply_algorithm is not supported')
            
        # Run completed
        if print_interval is not None: 
            print(f"Job completed with best GM: {self.best_y} at {self.best_id} that has survived last {self.best_trial} times of {self.global_structure_index} generations")
        
        if if_return_results:
            return self.pool_x, self.pool_y, self.pool_name
