'''
This file is an implementation for the well known ant colony optimization, Ant Colony Optimization is an algorithm that's
been using to resolve Travelling sales man problems and/or other problems within NP Hard domain like internet routing problem,
Scheduling problem and so on. The algorithm will proceed in the following steps
1. read the file that contains the dataset
2. Normalize the data
3. Create the ant colony
4. initialize the starting locations of the ant and initial pheromone level of each route
5. calculating the probabilities of ants from one city to another for all the cities, form a real number line such that
   all the probability of city travelled is on that line and since the summation of all the probability is 1, the range
   of the number line should be between 0 to 1
6. randomly generate a number from 0 to 1 and see which range in the number line it fits, then proceed the ant
7. update the pheromone level of each route
8. reset the ant to initial city
9. repeat the process from 7 to 10 until reaches num of iteration
'''
from Ant import Ant


ant_1 = Ant('Tommy')
ant_1.append_route_traveled_per_iteration('China')
ant_1.clear_route_traveled_per_iteration()
print(ant_1)
