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
import matplotlib.pyplot as plt


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


df = np.array(pd.read_csv("distance_of_26_cities.csv")) # minimum 937
# minimized route: [3, 5, 4, 6, 7, 8, 9, 13, 14, 12, 11, 10, 15, 18, 19, 17, 16, 20, 21, 25, 22, 23, 24, 0, 1, 2, 3]
df = data_process(df)
cities = []
for i in range(len(df[0])):
    cities.append(i)
cities = np.array(cities)


#  Parameters for the algorithm
POPULATION_SIZE = 100000
START_INDEX_FOR_CROSS_OVER = 10
END_INDEX_FOR_CROSS_OVER = 13
DEPRECATION_PERCENTAGE = 0.336
MUTATION_RATE = 0.1


# initialize a population object
population = Population(cities, POPULATION_SIZE, START_INDEX_FOR_CROSS_OVER, END_INDEX_FOR_CROSS_OVER, DEPRECATION_PERCENTAGE, MUTATION_RATE)

individual_that_is_child_list = []
num_of_epoch = []
fitness_values = []
total_individual_list = []



def main():
    population.initialize()
    epoch = 0
    total_individual = 0
    while population.size > 10:
        total_individual += population.size
        population.fitness(df)
        for j in list(population.fitness_value.values()):
            fitness_values.append(j)
        population.deprecate_population()
        population.partially_mapped_cross_over()
        population.mutate()

        population.fitness_value.clear()

        individual_that_is_child = 0
        for individual in population.population:
            if individual.is_child:
                individual_that_is_child += 1
        individual_that_is_child_list.append((1 - individual_that_is_child/population.size) * 100)
        epoch += 1
        num_of_epoch.append(epoch)
    return total_individual


counter = 0
for k in range(main()):
    total_individual_list.append(counter)
    counter += 1


# Plot the individual that is child
plt.figure(figsize=(5, 5))
plt.plot(num_of_epoch, individual_that_is_child_list)
plt.title('Relationship between num of child and epoch')
plt.xlabel('individual that is child(%)')
plt.ylabel('Epoch')
plt.grid(True)
plt.show()


plt.scatter(total_individual_list, fitness_values, c='red')
plt.title('Relationship between fitness value and individual')
plt.xlabel('individuals')
plt.ylabel('fitness value')
plt.grid(True)
plt.show()

print(population.most_fit_route)