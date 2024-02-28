'''
This file is a route class, it contains all the attributes and methods for a route
'''


class Route:
    def __init__(self, name, distance, pheromone_level):
        self.name = name
        self.distance = distance
        self.pheromone_level = pheromone_level
        self.ant_traveled_on_this_route_per_iteration = []

    def set_pheromone_level(self, new_pheromone_level):
        self.pheromone_level = new_pheromone_level

    def get_pheromone_level(self):
        return self.pheromone_level

    def append_ant_traveled_on_this_route_per_iteration(self, ant):
        self.ant_traveled_on_this_route_per_iteration.append(ant)

    def clear_ant_traveled_on_this_route_per_iteration(self):
        self.ant_traveled_on_this_route_per_iteration.clear()

    def __str__(self):
        return self.name + ', distance: ' + str(self.distance) + ', ' + 'pheromone level: ' + str(self.pheromone_level) + ', ' + 'ant traveled per iteration: '+ str(self.ant_traveled_on_this_route_per_iteration)