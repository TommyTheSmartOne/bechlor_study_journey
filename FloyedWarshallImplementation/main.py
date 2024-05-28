'''
This file contains implementation for Floyed Warshall algorithm. It is used to find all the shortest
path between any two vertices on a graph. The algorithm will proceed in the following steps
1. initiate matrix A0 such that the matrix will contain all the distance between any 2 vertices
    for self loop we will initiate the distance be 0(we assume all the vertex exists a self loop). for
    vertex that does not have a direct path to other we initiate as inf(it is a bidirectional graph
    thus not all the vertex has a direct path to another)
2. generate matrix AX from A(X-1) such that the Xth row, Xth column, diagonal elements remains the same
    for all the other empty cells, we compare values from A(X - 1) with the path such that with same initial
    vertex and destination city but with an intermediate vertex. in our case the intermediate vertex would
    be vertex X. whichever has smaller value, we fill the empty cell with that value
3. Repeat step 2 until X = n where n = num of vertex in the graph
'''
import numpy as np
import pandas as pd
import re


def appending_intermediate_values(current_intermediate_cell, matrix_a_x_minus_one):
    '''
    This function will append all the elements to the column based on current_intermediate_cells. if the
    current cell is 1 then the first row and column will be filled from previous matrix
    :param matrix_a_x_minus_one:
    :param current_intermediate_cell:
    :return:
    '''
    matrix_a_x = [[None] * 10 for x in matrix_a_x_minus_one] # initialize a matrix filled with None such that with size
    # 10 x 10
    for row in range(len(matrix_a_x_minus_one)):
        for col in range(matrix_a_x_minus_one[row].shape[0]):
            if row == current_intermediate_cell or col == current_intermediate_cell:
                matrix_a_x[row][col] = matrix_a_x_minus_one[row][col]
            elif row == col:
                matrix_a_x[row][col] = 0.0
    return np.array(matrix_a_x)


def compare_distance(matrix_a_x_minus_one, current_intermediate_cell, initial_vertex, destination_vertex):
    '''
    This function will compare distance between the same elements in AX and A(X-1) and return whichever is smaller
    :return:
    '''
    direct_path = matrix_a_x_minus_one[initial_vertex][destination_vertex]
    # this path can be visualize as initial --> destination
    path_include_intermediate_vertex = matrix_a_x_minus_one[initial_vertex][current_intermediate_cell] + \
                                       matrix_a_x_minus_one[current_intermediate_cell][destination_vertex]
    # this path can be visualize as initial --> intermediate --> destination
    if direct_path > path_include_intermediate_vertex:
        return path_include_intermediate_vertex
    else:
        return direct_path


def generate_next_matrix(current_intermediate_cell, matrix_a_x_minus_one):
    '''
    This function will generate A(x) from A(X-1) as stated in the docstring
    :return:
    '''
    matrix_a_x = appending_intermediate_values(current_intermediate_cell, matrix_a_x_minus_one)

    for row in range(matrix_a_x.shape[0]):
        for col in range(matrix_a_x.shape[1]):
            if matrix_a_x[row][col] is None:
                matrix_a_x[row][col] = compare_distance(matrix_a_x_minus_one, current_intermediate_cell, row, col)
    return np.array(matrix_a_x)


# This will be our initial matrix such that follows the pattern explained in the docstring
df = np.array(pd.read_csv("distance_of_10_cities.csv"))
matrix = df.copy()


def main():
    '''
    This is the main loop of the algorithm
    :return:
    '''
    global matrix
    for vertex in range(df.shape[0]):
        matrix = generate_next_matrix(vertex, matrix)



main()
matrix = pd.DataFrame(matrix)
print(matrix.to_string(index=False))