'''
This file contains an implementation of quick sort and it will proceed in the following steps
1. randomly select a pivot
2. move all the number that is larger than this number to its right and split into 2 subset, thus
    all the number that is larger than pivot is on the right and smaller or equal is on the left
3. repeat step 2 until then length of each subset is smaller than 2, we can now simply merge all
    the remaining number together.
Worst case: O(n^2)
Best case: O(nlog(n))
Quick sort rely on the core idea of sort each element on its correct position. The time complexity
depends on the choice of pivots.
'''
import numpy as np

def select_pivot(subset_arr):
    '''
    This function randomly select an index of subset_arr as pivot
    :return:
    '''
    return np.random.randint(len(subset_arr))


def partition(arr, low, high):
    '''
    partition the pivot and make sure the pivot will be on the right position in the end. will return the
    final position of the pivot
    :param arr:
    :param low:
    :param high:
    :return:
    '''
    pivot_index = low  # Select the pivot from the start index for simplicity
    pivot = arr[pivot_index]
    left = low + 1  # Start from the next element after pivot
    right = high

    while True:
        # Move left up to find an element greater than the pivot
        while left <= right and arr[left] <= pivot:
            left += 1

        # Move right down to find an element less than the pivot
        while left <= right and arr[right] >= pivot:
            right -= 1

        if left > right:
            break  # Indices have crossed, partitioning is done

        # Swap elements to move them to the correct side of the pivot
        arr[left], arr[right] = arr[right], arr[left]

    # Swap the pivot element with the right element to put pivot in correct position
    arr[low], arr[right] = arr[right], arr[low]
    return right



def quick_sort(arr, low, high):
    '''
    recursively partition both side of the arr
    :param arr:
    :param low:
    :param high:
    :return:
    '''
    if low > high:
        return
    split = partition(arr, low, high)
    quick_sort(arr, low, split - 1)
    quick_sort(arr, split + 1, high)


sample_arr = [3, 1, 5, 10, 7, 20, 2, 14]
quick_sort(sample_arr, 0, len(sample_arr) - 1)
print(sample_arr)