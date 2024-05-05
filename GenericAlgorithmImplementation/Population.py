'''
This file contains the class population.
'''

# Corresponding import for the file
import numpy as np
from tqdm import tqdm


class Population:
    def __init__(self, cities: np.ndarray, size: int, index_for_apply_cross_over: int):
        self.size = size
        self.cities = cities
        self.most_fit_route = {}  # this dictionary will record the most fit route with the key being distance,
        # the value being the route(a list)
        self.fitness_value = {}  # this is a dictionary contains key is the index for the population and the value is
        # the fitness value
        self.population = [None] * size
        self.index_for_apply_cross_over = index_for_apply_cross_over

    def shuffle_and_return(self, np_arr):
        np.random.shuffle(np_arr)
        return np_arr


    def initialize(self):
        '''
        This function is to initialize the populations, each individual in the population is a potential solution
        :return:
        '''
        for i in range(self.size):
            self.population[i] = self.shuffle_and_return(self.cities.copy())  # shuffle the route and append them in
            # the population.Thus each individual in the population is a solution


    def cross_over(self):
        '''
        This function will perform a cross over operations. That is
        :return:
        '''
        pbar = tqdm(total=len(self.population))
        counter = 0
        while counter < self.size:
            pbar.update(1)
            parent_list = []
            children = []
            for j in range(2):  # since there exists 2 individuals in a group of parents
                index_for_individuals_that_is_parent = np.random.randint(0, len(self.population))
                parent_list.append(index_for_individuals_that_is_parent)
            for k in range(self.index_for_apply_cross_over):
                children = self.population[parent_list[0]].copy()
                children[k] = self.population[parent_list[1]][k]
            self.population.append(children)  # append children back to the population
            counter += 1
        print(self.population)


    def fitness(self, df: [[]]):
        '''
        This function will compute the fitness for value for a individuals in the population.
        :return:
        '''
        # Since we are resolving TSP, the fitness value would be the total distance
        for i in range(self.size):
            temp = 0
            for j in range(self.population[i].size):
                temp += df[self.population[i][j]][self.population[i][j - 1]]
            self.fitness_value[i] = temp  # !!!!!!Important, after every epoch we must clear this dictionary


    def mutate(self):
        '''

        :return:
        '''

    # def update_population(self):
