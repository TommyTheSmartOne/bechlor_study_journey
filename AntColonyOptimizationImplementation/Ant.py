'''
This file is an ant class, it contains all the attributes and method for an ant
'''


class Ant:
    def __init__(self, name):
        self._name = name
        self._city_traveled_per_iteration = []
        self._location = 0

    def set_ant_location(self, location):
        self._location = location

    def get_location(self):
        return self._location

    def append_city_traveled_per_iteration(self, location):
        self._city_traveled_per_iteration.append(location)

    def clear_city_traveled_per_iteration(self):
        self._city_traveled_per_iteration.clear()

    def get_city_traveled_per_iteration(self):
        return self._city_traveled_per_iteration

    def get_name(self):
        return self._name

    def __str__(self):
        return 'name: ' + self._name + ', ' + 'route traveled: ' + str(self._city_traveled_per_iteration)