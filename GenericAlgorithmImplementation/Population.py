'''
This file contains the class population.
'''

# Corresponding import for the file
import numpy as np
from tqdm import tqdm
import math
from Individual import *


class Population:
    def __init__(self, cities: np.ndarray, size: int, start_index_for_apply_cross_over: int,
                 end_index_for_apply_cross_over: int, start_index_for_apply_mutate: int
                 , end_index_for_apply_mutate: int, deprecation_percentage: float, mutation_rate: float):
        self.size = size
        self.cities = cities
        self.most_fit_route = {
            math.inf: []}  # this dictionary will record the most fit route with the key being distance,
        # the value being the route(a list)
        self.fitness_value = {}  # this is a dictionary contains key is the index for the population and the value is
        # the fitness value
        self.population = [0] * size
        self.start_index_for_apply_cross_over = start_index_for_apply_cross_over
        self.end_index_for_apply_cross_over = end_index_for_apply_cross_over
        self.start_index_for_apply_mutate = start_index_for_apply_mutate
        self.end_index_for_apply_mutate = end_index_for_apply_mutate
        self.deprecation_percentage = deprecation_percentage
        self.mutation_rate = mutation_rate

    def shuffle_and_return(self, np_arr):
        np.random.shuffle(np_arr)
        return np_arr

    def initialize(self):
        '''
        This function is to initialize the populations, each individual in the population is a potential solution
        :return:
        '''
        for i in range(self.size):
            self.population[i] = Individual(list(
                self.shuffle_and_return(self.cities.copy())))  # shuffle the route and append them in
            self.population[i].route.append(self.population[i].route[0])  # For TSP we need to go back to the original route
            # the population.Thus each individual in the population is a solution
        # print(len(self.population))
        # print(self.size)
    def partially_mapped_cross_over(self):
        '''
        This function will perform a cross over operations. That is
        :return:
        '''

        odd_size = int(self.size // 2)
        pbar = tqdm(total=odd_size)
        counter = 0
        while counter < odd_size:
            pbar.update(1)
            parent_list = []
            for j in range(2):  # since there exists 2 individuals in a group of parents
                index_for_individuals_that_is_parent = np.random.randint(0, len(self.population))
                parent_list.append(index_for_individuals_that_is_parent)
                self.population[index_for_individuals_that_is_parent].update_is_parent()

            children = Individual([0] * (len(self.cities) + 1)) # initialize a children object

            for cities_in_children in range(self.start_index_for_apply_cross_over, self.end_index_for_apply_cross_over):
                children.route[cities_in_children] = self.population[parent_list[0]].route[cities_in_children]  # First append the
                # part we need to apply crossover.
                if self.population[parent_list[1]].route[cities_in_children] in self.population[parent_list[0]].route[
                                                                          self.start_index_for_apply_cross_over: self.end_index_for_apply_cross_over]:
                    continue
                    # the if statement will check if the value in cross over part in parent 1 is also
                    # appears in parent 0. if so we can skip this element since we just/will append this element to
                    # children


                index_for_cities_in_parent_1 = self.population[parent_list[1]].route.index(children.route[cities_in_children])
                # allocate the cities index in parent list 1. Since this index has been occupied in the previous
                # line of code. We now have to append the same index in parent list 1 to children by mapping.
                while index_for_cities_in_parent_1 in range(self.start_index_for_apply_cross_over,
                                                            self.end_index_for_apply_cross_over):
                    index_for_cities_in_parent_1 = self.population[parent_list[1]].route.index(
                        self.population[parent_list[0]].route[index_for_cities_in_parent_1])

                children.route[index_for_cities_in_parent_1] = self.population[parent_list[1]].route[cities_in_children]
                # here is a more specific example: say the part we are applying cross over is the middle list:
                # parent 0: [ 0, 1, 2, 3, [5, 7, 16, 9, 10], 4, 6, 8, 11]
                # parent 1: [ 2, 3, 6, 5, [1, 16, 8, 4, 11], 7, 9, 10, 0]
                # We first append all the cross over part in parent
                # 0 into the children. We will now notice the index for cross over in parent 1 has been occupied. we
                # now perform partial mapping. for every index that's been occupied, we find the index of the same
                # value in the rest part of parent 1(this index must be somewhere else instead in the cross over part.
                # unless for the value is equal). We then append this value to the children with the index we just
                # allocated. for instance. in parent 1, the first value is 1, but the index has already been occupied
                # now we go to the corresponding index in parent 0. which is 5, we then allocate the index of value 5 in
                # parent 1, which is 3. now in children index 3 we can append value 1... etc. In the end, we will obtain
                # a children with length 2 * len(part_apply_cross_over). we then just need to append the rest of parent
                # list 1 to the children with the same index. There is 2 special conditions which is why we have the if
                # statement and the while loop. however we can skip the explanation since it is too complex

            for remaining_cities_index in range(len(self.cities)):
                if self.population[parent_list[1]].route[remaining_cities_index] not in children.route:
                    children.route[remaining_cities_index] = self.population[parent_list[1]].route[remaining_cities_index]
            children.route[len(self.cities)] = children.route[0]

            self.population.append(children)  # append children back to the population
            self.size += 1
            counter += 1
        pbar.close()

    def fitness(self, df: [[]]):
        '''
        This function will compute the fitness for value for a individuals in the population.
        :return:
        '''

        # Since we are resolving TSP, the fitness value would be the total distance
        for i in range(self.size):
            temp = 0
            current_individual = self.population[i]
            for j in range(len(current_individual.route) - 1): # since last one is the original city, the distance between the
                # original city to itself is 0 so we don't have to worry about it
                temp += df[self.population[i].route[j]][self.population[i].route[j + 1]]
            self.fitness_value[i] = temp  # !!!!!!Important, after every epoch we must clear this dictionary
            if temp < list(self.most_fit_route.keys())[-1]:  # if the new fittest route distance is smaller than the old
                # distance
                self.most_fit_route.clear()  # clear the old fittest route
                self.most_fit_route[temp] = self.population[i].route  # append the new fittest route
        self.fitness_value = {k: v for k, v in
                              sorted(self.fitness_value.items(), key=lambda item: item[1])}  # sort the population
        # in ascending order based on the distance


    def mutate(self):
        '''
        This function will mutate a subset of the gene, the size of the subset are depends on the size of the gene, if
        the objective is only for a very small set of cities, we won't apply mutation.
        :return:
        '''
        for i in range(math.floor(self.size * self.mutation_rate)):
            index_of_individual_to_apply_mutate = np.random.randint(self.size)
            for j in range(self.start_index_for_apply_mutate, self.end_index_for_apply_mutate):
                temp = self.population[index_of_individual_to_apply_mutate].route[
                    j]  # Since this implementation is specify on TSP, we must not just simply flip
                # a digits like what normal generic algorithm does(Since in TSP each city must only be travel once).
                # We can perform a swap with some random indices.
                index_being_swap = np.random.randint(len(self.population[index_of_individual_to_apply_mutate].route) - 1)
                # excluding the last digit
                self.population[index_of_individual_to_apply_mutate].route[j] = \
                    self.population[index_of_individual_to_apply_mutate].route[
                        index_being_swap]  # generate a random index in the same
                # individual gene(city route) to swap.
                self.population[index_of_individual_to_apply_mutate].route[index_being_swap] = temp
            self.population[index_of_individual_to_apply_mutate].route[-1] = self.population[index_of_individual_to_apply_mutate].route[0]

    def deprecate_population(self):
        '''
        This function will update the size of the population along with deleting individuals that has the lower fitness
        values, the number of individual to deprecate is dependent on the deprecation percentage
        :return:
        '''

        route_being_deleted = list(self.fitness_value.keys())[
                              math.floor(self.size - self.size * self.deprecation_percentage): self.size]
        # obtain the further back indices based on the deprecation_percentage. These are the route/individual we will delete
        # since we sorted in ascending orders. Indicating the further back it is the 'less' fit it is(since the distance
        # would be longer)
        self.size -= len(route_being_deleted)
        for route_index in route_being_deleted:
            self.fitness_value.pop(route_index)  # delete the fitness value along with the route in the population.
        route_being_deleted = set(route_being_deleted)

        self.population = [item for idx, item in enumerate(self.population) if idx not in route_being_deleted]

