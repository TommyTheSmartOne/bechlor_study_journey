from Population import *
import numpy as np
import pandas as pd


df = np.array(pd.read_csv("distance_of_5_cities.csv"))
cities = []
for i in range(len(df[0])):
    cities.append(i)
cities = np.array(cities)


population = Population(cities, 100, 2)
population.initialize()
population.cross_over()
population.fitness(df)
print(population.fitness_value)
