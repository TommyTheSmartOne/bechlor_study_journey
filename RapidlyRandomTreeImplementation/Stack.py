'''
This file implements a stack
'''


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, data):
        self.stack.append(data)

    def pop(self):
        return self.stack.pop(-1)

    def peak(self):
        return self.stack[-1]

    def is_empty(self):
        if self.stack is None:
            return True
        return False

    def get_stack(self):
        return self.stack
