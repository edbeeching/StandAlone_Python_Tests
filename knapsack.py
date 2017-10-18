# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:17:39 2017

@author: Edward
"""
import numpy as np
num_objects = 100
poutlation_size = 100
# list of values
values = np.random.randint(1,20, size=(num_objects,))
# list of wieghts
weights = np.random.randint(1,100,size=(num_objects,))
# max capacity
max_capacity = 200
# potential solution
potential_solutions = np.random.binomial(1,0.5, size=(num_objects,100))
# evaluation function of a solution
def fitness(solution, values, weights, max_capacity):
    val = 0
    weight = 0
    for s,v,w in zip(solution, values, weights):
        if s == 1:
            if weight + w <= max_capacity:
                val += v
                weight += w
    return val


for sol in potential_solutions:
    print(fitness(sol, values, weights, max_capacity ))
