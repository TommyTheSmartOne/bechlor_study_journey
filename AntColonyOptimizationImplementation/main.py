'''
This file is an implementation for the well known ant colony optimization, Ant Colony Optimization is an algorithm that's
been using to resolve Travelling sales man problems and/or other problems within NP Hard domain like internet routing problem,
Scheduling problem and so on. The algorithm will proceed in the following steps
1. read the file that contains the dataset
2. Create the ant colony and routes
3. initialize the starting locations of the ant and initial pheromone level of each route
4. calculating the pheromone level of each routes and along with the desire of one ants to another city
5. calculating the probabilities of ants from one city to another for all the cities, form a real number line such that
   all the probability of city travelled is on that line and since the summation of all the probability is 1, the range
   of the number line should be between 0 to 1
6. randomly generate a number from 0 to 1 and see which range in the number line it fits, then proceed the ant
7. update the pheromone level of each route
8. reset the ant to initial city
9. repeat the process from 4 to 7 until reaches num of iteration
'''
from Ant import Ant
from City import City
import numpy as np
import pandas as pd
from tqdm import tqdm
import re as re
import sys

def data_process_for_42_cities(np_two_D_array):
    result_array = []
    for sub_np_array in np_two_D_array:
        sub_np_array = re.split('\s+', sub_np_array[0])
        sub_np_array.pop(0)
        sub_np_array = np.array(sub_np_array)
        sub_np_array = sub_np_array.astype(int).tolist()
        result_array.append(sub_np_array)
    return result_array


def randomly_generate_cities(N: int):
    "Make a set of n cities, each with random coordinates within a (width x height) rectangle."
    city_matrix = []
    b = np.random.randint(1, 100, size=(N, N))
    b_symm = (b + b.T) / 2
    counter = 0
    for i in range(len(b_symm)):
        b_symm[i][counter] = 0
        counter += 1
    return b_symm



# reading data from csv file and transform it to numpy array


def create_cities(initial_pheromone_level):
    '''
    This function is to initialize the number of cities the total graph contains, Note since the graph is bidirectional(
    or you may say undirected since both direction has the same distance, but here for simplicity in
    implementation, we say it is bidirectional with the same weight in both direction) thus we can create such a symmetric,
    diagonal matrix. Notice there will be a 'overlapping' in terms of the elements in the matrix
    :param initial_pheromone_level:
    :return:
    '''
    cities = []
    for city in range(0, len(df)):  # create cities depends on num given in df and append in a list
        cities.append(City(city))
    for city in range(0, len(cities)):
        for adjacent_cities in range(0, len(df[city])):
            if adjacent_cities != cities[city].get_name():
                cities[city].set_adjacent_city_distance_pheromone_level_dic(initial_pheromone_level,
                                                                            df[city][adjacent_cities],
                                                                            cities[adjacent_cities])

    return cities


def calc_desirability(distance):
    return 1 / distance


def calc_probability_from_one_city_to_adjacency_cities():
    '''
    This function is to calculate the probabilities from ini city to all of its adjacent cities such that the adjacent
    city haven't been traveled in this iteration.
    :param ini_city:
    :return:
    '''
    probability_list = {}
    total_multiple = 0
    for key in current_city.get_adjacent_city_distance_pheromone_level_dic().keys():
        if not key.get_is_traveled():  # if the given ant hasn't traveled this city yet
            total_multiple += calc_desirability(
                (current_city.get_adjacent_city_distance_pheromone_level_dic()[key][0]) ** ALPHA) \
                              * (current_city.get_adjacent_city_distance_pheromone_level_dic()[key][1] ** BETA)
            # refer to the formula
            probability_list[key.get_name()] = \
                (calc_desirability(current_city.get_adjacent_city_distance_pheromone_level_dic()[key][0]) ** ALPHA) \
                * (current_city.get_adjacent_city_distance_pheromone_level_dic()[key][1] ** BETA)
        # append each adjacent cities probability to a dictionary, where the key is the name of the route
        # and the value is the numerator

    for key_in_probability_list in probability_list:
        probability_list[key_in_probability_list] = probability_list[key_in_probability_list] / total_multiple
    # now we use the numerator divide by the summation of all the adjacent cities
    return probability_list


def form_Number_line():
    '''
    This function will form a number line from the previous number we obtained
    :return:
    '''
    range_list = []
    probability_list = []
    for key in probability_dict:
        probability_list.append(probability_dict[key])
    for i in range(0, len(probability_list)):
        point_on_num_line = 0
        for j in range(0, i + 1):  # plus 1 so we can include 1, but we will exclude 0
            point_on_num_line += probability_list[j]
        range_list.append(point_on_num_line)
    return range_list


def randomly_generate_num() -> int:
    '''
    This function will randomly generate a number thus we could plot in our number line and proceed the ants
    :return:
    '''
    num_generated = np.random.uniform(0.0, 1.0)
    for point_index in range(0, len(number_line)):
        if num_generated <= number_line[point_index]:
            return point_index


def proceed_the_ant():
    '''
    This function we proceed the ants, meanwhile we do some updates on ant object
    :return:
    '''
    ant.set_ant_location(destination_city_index)
    ant.append_city_traveled_per_iteration(destination_city.get_name())
    destination_city.set_traveled_state(True)
    destination_city.set_ant_traveled()


def update_pheromone_on_one_route(new_pheromone_level, desti_city: City, ini_city: City):
    ini_city.modify_adjacent_city_pheromone_level(new_pheromone_level, desti_city)


def calc_pheromone_level(key: City) -> float:
    pheromone_level = (
                              1 - EVAPORATION_COEFICIENT) * INI_PHEROMONE_LEVEL + key.get_ant_traveled() * PHEROMONE_ANT_RELEASED
    return pheromone_level


def calc_iteration_distance():
    path = ant.get_city_traveled_per_iteration()
    distance = 0
    for i in range(0, len(path)):
        if i != len(path) - 1:
            distance += df[path[i]][path[i + 1]]
        else:
            distance += df[path[i]][path[0]]
    return distance


def compare_iterations():
    global current_iteration_distance, best_iteration_distance, best_iteration_path
    current_iteration_distance = calc_iteration_distance()
    print('current iteration distance: ', current_iteration_distance, 'best iteration distance: ', best_iteration_distance)
    if current_iteration_distance < best_iteration_distance or best_iteration_distance == 0:
        best_iteration_path = ant.get_city_traveled_per_iteration().copy()
        best_iteration_distance = current_iteration_distance


# Here input the data of cities, for the format of the data see example "distance_of_5_cities"
# df = np.array(pd.read_csv("distance_of_42_cities.csv"))
# df = randomly_generate_cities(50)
# df = data_process_for_42_cities(df)
df = np.array(pd.read_csv("distance_of_5_cities.csv"))


# constant term
PHEROMONE_ANT_RELEASED = 1
EVAPORATION_COEFICIENT = 0.6  # This num must be greater than 0 but smaller than 1, the larger it is the more
# pheromone evaporate per iteration
ITERATION_NUM = 1500
INI_PHEROMONE_LEVEL = 0.2
counter = 0
ALPHA = 10
BETA = 7.5

ant = Ant('Karolina')
city_colony = create_cities(INI_PHEROMONE_LEVEL)
best_iteration_path = []
best_iteration_distance = 0
current_iteration_distance = 0
pbar = tqdm(total=ITERATION_NUM)
while counter < ITERATION_NUM:
    for ants in range(0, len(df)):
        ini_city = np.random.randint(5)
        current_city = city_colony[ini_city]
        # current_city = city_colony[ants]
        current_city.set_traveled_state(True)
        ant.clear_city_traveled_per_iteration()
        # The following code is city select by an ant
        ant.append_city_traveled_per_iteration(current_city.get_name())
        for city_visited in range(0, len(df) - 1):
            probability_dict = calc_probability_from_one_city_to_adjacency_cities()
            number_line = form_Number_line()
            probability_dict_index = randomly_generate_num()
            destination_city_index = list(probability_dict.keys())[probability_dict_index]
            destination_city = city_colony[destination_city_index]
            proceed_the_ant()
            current_city = city_colony[ant.get_location()]  # after ant proceed, the ant city location should be update
        destination_city = city_colony[ini_city]  # after the entire iteration we must tell the ants to go back to the
        # ini_city
        proceed_the_ant()
        compare_iterations()
        for cities in city_colony:  # clear data in cities so we can start a new iteration
            cities.set_traveled_state(False)
    # after one colony of ants proceed we now update pheromone level
    for cities in city_colony:
        for keys in list(cities.get_adjacent_city_distance_pheromone_level_dic().keys()):
            update_pheromone_on_one_route(calc_pheromone_level(keys), keys, cities)
        cities.clear_ant_traveled()
    counter += 1
    pbar.update(1)
pbar.close()
print("path best iteration: " + str(best_iteration_path))
print("best iteration distance: " + str(best_iteration_distance))
