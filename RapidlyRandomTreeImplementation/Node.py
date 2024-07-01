class Node:
    def __init__(self, x, y, identity, parent):
        self.pos = [x, y]
        self.identity = identity
        self.parent = parent


    def __str__(self):
        return self.identity



