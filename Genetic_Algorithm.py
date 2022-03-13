#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:44:52 2021

@author: hariprasadrajendran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:49:39 2021

@author: hariprasadrajendran
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import datetime

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from k_means_constrained import KMeansConstrained

##Open Saved Model


filename="XGBoost_model.sav"

Saved_model = pickle.load(open(filename, 'rb'))


## Testing Locations

Office = {'Of': (40.703761, -73.886496)}

test_locations = {
                  'L1': (40.819688, -74.515091),
                  'L2': (41.785461, -74.643621),
                  'L3': (40.764198, -73.910785),
                  'L4': (41.198790, -73.953285),
                  'L5': (40.734851, -74.152950),
                  'L6': (40.473613, -74.327998),
                  'L7': (40.745313, -73.993793),
                  'L8': (40.815421, -73.941761),
                  'L9': (41.346929, -73.846295),
                  'L10': (40.768790, -73.953285)
             }

## Testing date

test_date = [14, 5, 2016]
year_i = int(test_date[2])
month_i = int(test_date[1])
day_i =  int(test_date[0])

i_date = datetime.date(year_i, month_i, day_i)

print(i_date.year)

from copy import copy
def input_list(points):
    shuffle = copy(points[1:])
    np.random.shuffle(shuffle)
    shuffle.append(points[0])
    shuffle.insert(0,points[0])
    return list(shuffle)

    
shuffle_location = input_list(list(test_locations.keys()))

print(shuffle_location)
#### combinations of locations####

def create_generation(points, population=100):
    generations = [input_list(points) for _ in range(population)]
    return generations

generation_combos = create_generation(list(test_locations.keys()),population=10)
print(generation_combos)

## trip duration calculations based on a XGBoost predictive model

def tripDuration_between_points(id1, id2,hour, date, passenger_count = 1, pickup_minute = 0):
   
    latitude_pick = np.radians(id1[0])
    latitude_drop = np.radians(id2[0])
    latitude_diff = np.radians(id2[0] - id1[0])
    longitude_diff = np.radians(id1[1] - id1[1])
    p1 = np.sin(latitude_diff/2)**2 + np.cos(latitude_pick) * np.cos(latitude_drop) * np.sin(longitude_diff/2)**2
    q1 = 2 * np.arctan2(np.sqrt(p1), np.sqrt(1-p1))
   
    # input_data = {#'passenger_count': passenger_count,
    #               'pickup_longitude': id1[1], 'pickup_latitude': id1[0],
    #               'dropoff_longitude': id2[1], 'dropoff_latitude': id2[0],
    #               'Year': date.year,
    #               'pickup_month': date.month, 'pickup_date': date.day,
    #               'pickup_hour': hour, 'pickup_minutes': pickup_minute,
    #               'Trip_Distance': q1*3956} 

    # table = pd.DataFrame([input_data], columns=input_data.keys())
    # result = np.exp(Saved_model.predict(xgb.DMatrix((table)))) - 1
    
    input_data = {#'passenger_count': passenger_count,
                  'pickup_longitude': id1[1], 'pickup_latitude': id1[0],
                  'dropoff_longitude': id2[1], 'dropoff_latitude': id2[0],
                  'Year': date.year,
                  'pickup_month': date.month, 'pickup_date': date.day,
                  'pickup_hour': hour, 'pickup_minutes': pickup_minute,
                  'Trip_Distance': q1*3956} 

    table = pd.DataFrame([input_data], columns=input_data.keys())
    result = np.exp(Saved_model.predict(table)) - 1
    return result[0]                                                            
    
coordinates = test_locations
def Off_Off_distance(value):
    dis=0
   
    for i, point_id in enumerate(value[:-1]):
       # print(coordinates[point_id],coordinates[value[i+1]])
        dis = dis + tripDuration_between_points(coordinates[point_id], coordinates[value[i+1]], 11, i_date)
    return dis

def check_fitness(combi):
    """
    Goes through every guess and calculates the fitness score. 
    Returns a list of tuples: (guess, fitness_score)
    """
   
    fitness_indicator = []
    for guess in combi:
        
        fitness_indicator.append((guess, Off_Off_distance(guess)))
    return fitness_indicator

print(check_fitness(generation_combos))


## Selection ####
## Selecting parents from the combos to create the next generation ###
### selection based on fitness : one with the lowest score( shortest time) is ranked first ###
### first rand(n) best populations are selected and then appended with the rest population(randomnly)  ####

def selection_from_generation(cur_gen, N=10, rand=5, verbose=False, mutation_rate=0.08):
    ## Get the best population from the last step
    
    fittness_scores = check_fitness(cur_gen)
    ## sort such that lowest trip duration is first
    sorted_cur_gen = sorted(fittness_scores, key=lambda x: x[1]) 
   #print(sorted_cur_gen[0:20])
    selectionResults = [x[0] for x in sorted_cur_gen[:N]]
    #print("################")
    #print(selectionResults)
    best_guess = selectionResults[0]
    
    if verbose:
        # If we want to see what the best current guess is!
        print(best_guess)
    
    for _ in range(rand):
        xrand = np.random.randint(len(cur_gen)) # returns random numbers from 0-len(cur_gen)
        selectionResults.append(cur_gen[xrand])

    np.random.shuffle(selectionResults)
    return selectionResults, best_guess

## create child to include all locattions exactly once using ordered crosover
## subset of parent1 is randomly selected and remaining is filled with genes from parent2
import random 

def breed(parent1, parent2):
    # print("Parents")
    # print(parent1, parent2)
    child=np.zeros(len(parent1))
    child_from_parent1 = []
    child_from_parent2 = []
    
    # p = int(random.uniform(0.0,1.0)*len(parent1))
    # q = int(random.uniform(0.0,1.0)*len(parent1))
    
    # start = min(p,q)
    # end = max(p,q)
    
    for h in range(0, len(parent1)//2):
        child_from_parent1.append(parent1[h])
        
    child_from_parent2 = [item for item in parent2 if item not in child_from_parent1]
    child_from_parent2.append(child_from_parent1[0])
    #print(child_from_parent1, child_from_parent2)
    child = child_from_parent1 + child_from_parent2
    
    return child

# def breed(parent1, parent2):
#     """ 
#     Take some values from parent 1 and hold them in place, then merge in values
#     from parent2, filling in from left to right with cities that aren't already in 
#     the child. 
#     """
#     #list_of_ids_for_parent1 = list(np.random.choice(parent1, replace=False, size=len(parent1)//2))
#     #print(list_of_ids_for_parent1)
#     #print(type(parent1[0]))
#     child = [11 for _ in parent1]
    
#     for ix in range(0, len(parent1)//2):
#         child[ix] = parent1[ix]
#         print(child)
#     for z, val in enumerate(child):
#         if val == 11:
#             for val2 in parent2:
#                 if val2 not in child:
#                     print(val2)
#                     child[z] = val2
#                     break
#     child[-1] = child[0]
#     return child

    
def make_children(old_gen, perCouple_children=1):
   
    mid_point = len(old_gen)//2
    next_generation = [] 
    
    for b, parent in enumerate(old_gen[:mid_point]):
        for _ in range(perCouple_children):
            next_generation.append(breed(parent, old_gen[-b-1]))
    return next_generation  

# present_generation = create_generation(list(test_locations.keys()), population=500)
# displaying_nGen = 5

# for k in range(50):
#     if k%displaying_nGen == 0:
#         print('Generation {}: {}'.format(k,len(present_generation)))
#         is_verbose = True
#     else:
#         is_verbose = False
    
#     selections, best_guess = selection_from_generation(present_generation, N=250, rand=100, verbose=is_verbose)
#     present_generation = make_children(selections, children_per_couple=3)

def check_fitness_a(combi):
   
    fitness_indicator = []
    fitness_indicator.append((combi, Off_Off_distance(combi)))
    return fitness_indicator



def final_solve(cur_generation, max_generations, take_best_N, take_random_N,
                    mutation_rate, perCouple_children, print_every_n_generations, verbose=False):
    
    fitness_tagging = []
    for i in range(max_generations):
        if verbose and not i % print_every_n_generations and i > 0:
            print("Generation %i: "%i, end='')
            print(len(cur_generation))
            print("Current Best Score: ", fitness_tagging[-1])
            is_verbose = True
        else:
           
            is_verbose = False
        parents, best_guess = selection_from_generation(cur_generation, N=take_best_N, rand=take_random_N, 
                                                            verbose=is_verbose, mutation_rate=mutation_rate)
        
        fitness_tagging.append(check_fitness_a(best_guess))
        current_generation = make_children(parents, perCouple_children=perCouple_children)
    
    return fitness_tagging, best_guess

if len(test_locations)<=6:
    Office.update(test_locations)
    coordinates = Office      
    current_generation = create_generation(list(Office.keys()),population=500)
    fitness_tagging, best_guess = final_solve(current_generation, 100, 150, 70, 0.5, 3, 5, verbose=True)

else:
    df = pd.DataFrame.from_dict(test_locations, orient='index')
    ckm = KMeansConstrained(n_clusters=2,size_min=1,size_max=6, init='k-means++')
    ckm.fit(df[df.columns[0:2]])
    centers = ckm.cluster_centers_ # Coordinates of cluster centers.
    labels = ckm.predict(df[df.columns[0:2]]) # Labels of each point
    df['cluster'] = labels
    print(df)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap="Paired")

    set1 = df.loc[df['cluster']==1]
    set1 = set1.iloc[:,0:2]
    set1.rename_axis("newhead").reset_index()
    set1 = {x[0]: x[1:] for x in set1.itertuples(index=True)}

    set2 = df.loc[df['cluster']==0]
    set2 = set2.iloc[:,0:2]
    set2.rename_axis("newhead").reset_index()
    set2 = {x[0]: x[1:] for x in set2.itertuples(index=True)}

    Office = {'Of': (40.703761, -73.886496)}
    Office.update(set1)
    coordinates = Office      
    current_generation = create_generation(list(Office.keys()),population=500)
    fitness_tagging, best_guess = final_solve(current_generation, 100, 150, 70, 0.5, 3, 5, verbose=True)
    
    Office = {'Of': (40.703761, -73.886496)}
    Office.update(set2)
    coordinates = Office      
    current_generation = create_generation(list(Office.keys()),population=500)
    fitness_tagging, best_guess = final_solve(current_generation, 100, 150, 70, 0.5, 3, 5, verbose=True)



## Without genetic Algorithm ###
import itertools
if len(test_locations)<=6:
    Office.update(test_locations)
    coordinates = Office 
    sut = input_list(list(Office.keys()))
    print(sut)
    tus = Off_Off_distance(sut)
    print("Distance without optimization: {}".format(tus))

else:
    test1 = dict(itertools.islice(test_locations.items(), 6))
    Office = {'Of': (40.703761, -73.886496)}
    Office.update(test1)
    coordinates = Office 
    sut1 = input_list(list(Office.keys()))
    print(sut1)
    tus1 = Off_Off_distance(sut1)
    print("TRIP DURATION of Route 1 without optimization: {}".format(tus1))

    test2 = {k: test_locations[k] for k in list(test_locations)[6:]}
    Office = {'Of': (40.703761, -73.886496)}
    Office.update(test2)
    coordinates = Office 
    sut2 = input_list(list(Office.keys()))
    print(sut2)
    tus2 = Off_Off_distance(sut2)
    print("TRIP DURATION of Route 2 without optimization: {}".format(tus2))


 


# Office = {'Of': (40.703761, -73.886496)}
# Office.update(test_locations)
# coordinates = Office 
# res = Off_Off_distance(list(coordinates))
# print("Distance of Route 1 without optimization: {}".format(res))



































