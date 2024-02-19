'''
This file is an ant class, it contains all the attributes and method for an ant
'''
class Ant:
    def __init__(self, name):
        self._name = name
        self._route_traveled_per_iteration = []

    def append_route_traveled_per_iteration(self, location):
        self._route_traveled_per_iteration.append(location)

    def clear_route_traveled_per_iteration(self):
        self._route_traveled_per_iteration.clear()

    def get_route_traveled_per_iteration(self):
        return self._route_traveled_per_iteration

    def get_name(self):
        return self._name

    def __str__(self):
        return self._name + '' + str(self._route_traveled_per_iteration)