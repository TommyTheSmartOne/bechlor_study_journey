'''
This file contains the class population.
'''

# Corresponding import for the file
import numpy as np
from tqdm import tqdm
import math


class Population:
    def __init__(self, cities: np.ndarray, size: int, index_for_apply_cross_over: int, start_index_for_apply_mutate: int
                 , end_index_for_apply_mutate: int, number_of_mutation_per_epoch: int, deprecation_percentage: float):
        self.size = size
        self.cities = cities
        self.most_fit_route = {
            math.inf: []}  # this dictionary will record the most fit route with the key being distance,
        # the value being the route(a list)
        self.fitness_value = {}  # this is a dictionary contains key is the index for the population and the value is
        # the fitness value
        self.population = [None] * size
        self.index_for_apply_cross_over = index_for_apply_cross_over
        self.start_index_for_apply_mutate = start_index_for_apply_mutate
        self.end_index_for_apply_mutate = end_index_for_apply_mutate
        self.number_of_mutation_per_epoch = number_of_mutation_per_epoch
        self.deprecation_percentage = deprecation_percentage

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
            if temp < list(self.most_fit_route.keys())[-1]:  # if the new fittest route distance is smaller than the old
                # distance
                # self.most_fit_route.clear()  # clear the old fittest route
                self.most_fit_route[temp] = self.population[i]  # append the new fittest route

    def mutate(self):
        '''
        This function will mutate a subset of the gene, the size of the subset are depends on the size of the gene, if
        the objective is only for a very small set of cities, we won't apply mutation.
        :return:
        '''
        for i in range(self.number_of_mutation_per_epoch):
            index_of_individual_to_apply_mutate = np.random.randint(self.size)
            for j in range(self.start_index_for_apply_mutate, self.end_index_for_apply_mutate):
                temp = self.population[index_of_individual_to_apply_mutate][
                    j]  # Since this implementation is specify on TSP, we must not just simply flip
                # a digits like what normal generic algorithm does(Since in TSP each city must only be travel once).
                # We can perform a swap with some random indices.
                index_being_swap = np.random.randint(self.population[index_of_individual_to_apply_mutate].size)
                self.population[index_of_individual_to_apply_mutate][j] = \
                    self.population[index_of_individual_to_apply_mutate][
                        index_being_swap]  # generate a random index in the same
                # individual gene(city route) to swap.
                self.population[index_of_individual_to_apply_mutate][index_being_swap] = temp

    def deprecate_population(self):
        '''
        This function will update the size of the population along with deleting individuals that has the lower fitness
        values, the number of individual to deprecate is dependent on the deprecation percentage
        :return:
        '''
        self.fitness_value = {k: v for k, v in
                              sorted(self.fitness_value.items(), key=lambda item: item[1])}  # sort the population
        # in ascending order based on the distance
        route_being_deleted = list(self.fitness_value.keys())[
                              math.floor(self.size - self.size * self.deprecation_percentage): self.size]
        self.size -= len(route_being_deleted)
        for route_index in route_being_deleted:
            self.fitness_value.pop(route_index) # delete the fitness value along with the route in the population.
            self.population.pop(route_index)
