'''
This file is a route class, it contains all the attributes and methods for a route
'''
from __future__ import annotations


class City:
    def __init__(self, name: int):
        self._name = name
        self._adjacent_city_distance_pheromone_level_dic = {}
        self._is_traveled = False
        self._ant_traveled = 0

    def set_ant_traveled(self):
        self._ant_traveled += 1

    def clear_ant_traveled(self):
        self._ant_traveled = 0

    def get_ant_traveled(self):
        return self._ant_traveled

    def set_adjacent_city_distance_pheromone_level_dic(self, pheromone_level, distance, destination_city: City):
        temp_list = [distance, pheromone_level]
        self._adjacent_city_distance_pheromone_level_dic[destination_city] = temp_list


    def modify_adjacent_city_pheromone_level(self, pheromone_level, destination_city: City):
        self._adjacent_city_distance_pheromone_level_dic[destination_city] = [self._adjacent_city_distance_pheromone_level_dic[destination_city][0]
            , pheromone_level]

    def set_traveled_state(self, val):
        self._is_traveled = val

    def get_is_traveled(self):
        return self._is_traveled

    def get_adjacent_city_distance_pheromone_level_dic(self):
        return self._adjacent_city_distance_pheromone_level_dic

    def get_name(self):
        return self._name

    def __str__(self):
        return self._name + ', Adjacent cities and corresponding distance and pheromone level: ' \
               + str(self._adjacent_city_distance_pheromone_level_dic)
