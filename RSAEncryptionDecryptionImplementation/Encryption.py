'''
This file will contain the process of encryption
'''


class Encryption:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.key = []

    def num_of_compute_co_prime(self):
        return (self.p - 1) * (self.q - 1)

    def is_even(self, num):
        '''
        In this function we will compute if given number is odd or even
        :param num:
        :return:
        '''
        if num % 2 == 0:
            return True
        else:
            return False

    def euclidean_algorithm_for_co_prime(self, x, y):
        remainder = 1  # initial remainder
        CD = [None, None]  # a list contain all the common divisor
        while True:
            if remainder == 0 and CD[-2] == 1:  # We will check if the greatest common divisor are 1. If so, we say x and
                # y are co-prime
                return True
            elif remainder == 0 and CD[
                -2] != 1:  # if the greatest common factor/divisor are not 1 then the two numbers
                # are not co-prime with each other
                return False
            remainder = x % y
            CD.append(remainder)
            x = y
            y = remainder


    def encrypt(self, message):
        '''
        This function will return the encrypted message, AKA public key
        :return:
        '''

        n = self.p * self.q
        phi = self.num_of_compute_co_prime()
        parity_of_n = self.is_even(n)
        parity_of_phi = self.is_even(
            phi)  # this two line we will check if n or phi is odd or even, by doing such we can
        # eliminate some checks in our following steps, note this is not in the original RSA encryption. A example would
        # be if n or phi is indeed even. Then when we check co-prime we can skip the even number since those number can
        # also divide by 2 and thus the number must not be co prime with both of them
        for e in range(2, phi):
            if (e % 2 == 0) and (parity_of_n or parity_of_phi):
                continue
            if self.euclidean_algorithm_for_co_prime(n, e) and self.euclidean_algorithm_for_co_prime(phi, e):
                # if both number and i has the greatest common factor 1 then i is co-prime with n and co-prime with phi
                self.key.append(e)
                self.key.append(n)
                cipher_text = (message ** e) % n
                return cipher_text
