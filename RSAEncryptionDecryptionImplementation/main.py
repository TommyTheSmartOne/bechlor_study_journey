'''
This file is the main such that calls methods from encryption and decryption to mimic an message transmission process
'''
from tqdm import tqdm
import numpy as np

def sieve_of_Eratosthenes(termination_index: int):
    '''
    This function used an ancient method to generate all the prime number from 2 up to the termination index. function
    will return a  generator
    :param termination_index:
    :param list_that_is_not_prime:
    :param p:
    :return:
    '''
    p = 2
    list_that_is_not_prime = []
    while p < termination_index:
        if p in list_that_is_not_prime:
            p += 1
            continue
        else:
            yield p
            for i in range(p ** 2, termination_index):
                if i % p == 0:
                    list_that_is_not_prime.append(i)
        p += 1


def p_q_generator():
    counter = 0
    for prime_number in sieve_of_Eratosthenes(100000):
        if np.random.rand() < 0.001 and counter != 2:
            yield prime_number
            counter += 1
        elif counter == 2:
            return


def prime_generator():
    for i in p_q_generator():
        print(i)


p = 24107
q = 29101

