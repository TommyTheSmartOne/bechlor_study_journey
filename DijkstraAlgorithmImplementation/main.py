'''
This file contains an implementation of Dijkstra algorithm which will find the shortest distance between any
2 vertices in a graph
The algorithm will proceed in the following steps(since the dataset we are using is preprocessed for this algorithm
we can skip the initialization process, that is, the city that is not directly connected we set the distance to
infinity, the cities distance to itself(self-loop) is set to 0. If we use any other dataset to test the distance in
the future, the dataset must be process first such that match the above description first).
1. sort the direct path distance between the initial cities and the destination cities in ascending orders
2. pick the next vertex with the shortest distance,
'''

# this dictionary will store all the relaxed value (shortest distance from source node) from source node to
# destination node with key being the vertex and value
relaxed_dict = {}


