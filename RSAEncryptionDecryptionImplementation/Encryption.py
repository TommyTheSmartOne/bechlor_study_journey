'''
This file will contain the process of encryption
'''


class Encryption:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.public_key = []

    def num_of_compute_co_prime(self):
        return (self.p - 1) * (self.q - 1)


    def greatest_common_divisor(self, x, y):
        while y != 0:
            (x, y) = (y, x % y)
        return x


    def encrypt(self, message):
        '''
        This function will return the encrypted message, AKA public key
        :return:
        '''
        n = self.p * self.q
        phi = self.num_of_compute_co_prime()
        e = 65537
        self.public_key.append(e)
        self.public_key.append(n)
        cipher_text = (message ** e) % n
        return self.public_key, cipher_text, phi
