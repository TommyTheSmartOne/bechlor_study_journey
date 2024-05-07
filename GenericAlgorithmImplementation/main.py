'''
The following file contains an implementation for TSP. All the information related to the population will be presented
in the population.py file. Note for different number of cities we need to apply parameter tuning. The file will be
implemented in the following sequence
1. process data
2. initialize population(note each individual in the population is a potential solutions)
3. Set the stopping condition and parameters(in our case the epoch will stop when population is smaller enough since the
   population is constantly decreasing due to the parameter being set)
4. start the epochs, for each epoch the following will proceed:
    a. compute the fitness value
    b. deprecate the one with lower fitness value
    c. apply cross_over/mating within the survivals and append the child back into the population
    d. mutate the gene(solutions) for certain individuals
'''

from Population import *
import numpy as np
import pandas as pd
import re as re


def data_process(np_two_D_array):
    '''
    This function is to normalize the data
    :param np_two_D_array:
    :return:
    '''
    result_array = []
    for sub_np_array in np_two_D_array:
        sub_np_array = re.split('\s+', sub_np_array[0])
        sub_np_array.pop(0)
        sub_np_array = np.array(sub_np_array)
        sub_np_array = sub_np_array.astype(int).tolist()
        result_array.append(sub_np_array)
    return result_array


df = np.array(pd.read_csv("distance_of_26_cities.csv"))
df = data_process(df)
cities = []
for i in range(len(df[0])):
    cities.append(i)
cities = np.array(cities)


#  Parameters for the algorithm
POPULATION_SIZE = 500000
START_INDEX_FOR_CROSS_OVER = 10
END_INDEX_FOR_CROSS_OVER = 16
START_INDEX_FOR_MUTATE = 4
END_INDEX_FOR_MUTATE = 10
DEPRECATION_PERCENTAGE = 0.5
MUTATION_RATE = 0.001


# initialize a population object
population = Population(cities, POPULATION_SIZE, START_INDEX_FOR_CROSS_OVER, END_INDEX_FOR_CROSS_OVER, START_INDEX_FOR_MUTATE,
                        END_INDEX_FOR_MUTATE, DEPRECATION_PERCENTAGE, MUTATION_RATE)


def main():
    population.initialize()
    while population.size > 1000:
        population.fitness(df)
        population.deprecate_population()
        population.partially_mapped_cross_over()
        population.mutate()
        print(population.size)
        print(population.most_fit_route)


main()




