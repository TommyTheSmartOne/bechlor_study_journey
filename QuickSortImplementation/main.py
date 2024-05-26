'''
This file contains an implementation of quick sort and it will proceed in the following steps
1. randomly select a pivot
2. move all the number that is larger than this number to its right and split into 2 subset, thus
    all the number that is larger than pivot is on the right and smaller or equal is on the left
3. repeat step 2 until then length of each subset is smaller than 2, we can now simply merge all
    the remaining number together.
'''
import numpy as np

def select_pivot(subset_arr):
    '''
    This function randomly select an index of subset_arr as pivot
    :return:
    '''
    return np.random.randint(len(subset_arr))


def split(subset_arr):
    '''
    This function sort all the larger number compare to pivot to its right and smaller to its left
    :return:
    '''
    pivot = select_pivot(arr)



arr = [3, 5, 7, 9, 10, 2, 1, 40]