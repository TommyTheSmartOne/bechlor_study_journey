'''
This file contains individual. The purpose of it is mainly for testing and debugging
'''
class Individual:
    def __init__(self, route: list):
        self.route = route
        self.parents = {}
        self.children = {}
        self.is_mutated = False
        self.is_child = False
        self.is_parent = False

    def is_child(self):
        return self.is_child

    def is_parent(self):
        return self.is_parent

    def is_mutated(self):
        return self.is_mutated

    def update_is_parent(self):
        '''
        The following 2 function along with this one is for update the state for this individual. Since there is no way
        to reverse these settings(ie. an individual can not unparent itself) We won't need another function to reverse
        this process
        :return:
        '''
        self.is_parent = True

    def update_is_child(self):
        self.is_parent = True

    def update_is_mutate(self):
        self.is_mutated = True

    def update_parents(self, parent_0: list, parent_1: list):
        self.parents['parent_0'] = parent_0
        self.parents['parent_1'] = parent_1
        self.update_is_child()

    def update_children(self, child: list):
        self.children['children'] = child
        self.update_is_parent()


