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
import numpy as np
import pandas as pd


df = np.array(pd.read_csv("distance_of_6_cities_one_way_graph.csv"))
print(df)




def relaxation(current_destination_distance, current_source_distance,
               distance_from_current_source_to_current_destination) -> int:
    '''
    This function will perform relaxation operation
    :return:
    '''
    if current_source_distance + distance_from_current_source_to_current_destination < current_destination_distance:
        return current_source_distance + distance_from_current_source_to_current_destination
    else:
        return current_destination_distance

def main() -> None:





source_node = 0
destination_node = 5
main()