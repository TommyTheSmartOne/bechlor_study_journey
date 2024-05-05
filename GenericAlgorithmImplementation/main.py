from Population import *
import numpy as np
import pandas as pd


df = np.array(pd.read_csv("distance_of_5_cities.csv"))
cities = []
for i in range(len(df[0])):
    cities.append(i)
cities = np.array(cities)


population = Population(cities, 100, 2, 0, 1, 2, 0.6)
population.initialize()
population.cross_over()
population.fitness(df)
population.mutate()
population.deprecate_population()
# print(population.fitness_value)
# print(population.most_fit_route)

