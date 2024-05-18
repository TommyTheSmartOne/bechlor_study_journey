'''
This file is the main such that calls methods from encryption and decryption to mimic an message transmission process
'''
from tqdm import tqdm
import numpy as np
from Encryption import *
from Decryption import *


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
    '''
    This generator will generate 2 prime number
    :return:
    '''
    counter = 0
    result = []
    for prime_number in sieve_of_Eratosthenes(100000): # call sieve_of_Eratosthenes generator to allocate all the prime
        # within the given limits
        if np.random.rand() < 0.001 and counter != 2:  # we only need 2 prime number
            result.append(prime_number)
            counter += 1
            if counter == 2:
                return result[0], result[1]




p, q = p_q_generator()
message = 2

encoder = Encryption(p, q)
public_key, cipher_text, phi = encoder.encrypt(2)
# Note in real world scenario phi is not public. Since here we will need to use e to calculate the key for decryption
# and thus we will just show.
decoder = Decryption(public_key, cipher_text, phi)
decrypted_message = decoder.decrypt()
