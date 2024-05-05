from Population import *
import numpy as np
import pandas as pd
import re as re


def data_process(np_two_D_array):
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

population = Population(cities, 50000, 8, 10, 13, 10, 0.6)
while population.size > 100:
    population.initialize()
    population.fitness(df)
    population.deprecate_population()
    population.cross_over()
    population.mutate()




