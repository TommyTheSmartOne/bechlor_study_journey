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
population_size = 500000
start_index_for_apply_cross_over = 10
end_index_for_apply_cross_over = 16
start_index_for_apply_mutate = 4
end_index_for_apply_mutate = 10
deprecation_percentage = 0.5


# initialize a population object
population = Population(cities, population_size, start_index_for_apply_cross_over, end_index_for_apply_cross_over, start_index_for_apply_mutate,
                        end_index_for_apply_mutate, deprecation_percentage)

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




