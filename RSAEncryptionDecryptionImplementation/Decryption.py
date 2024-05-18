class Decryption:
    def __init__(self, public_key: list, cipher_text, phi):
        self.cipher_text = cipher_text
        self.public_key = public_key
        self.d = self.compute_decryption_key(public_key[0], phi)

    def compute_decryption_key(self, e, phi):
        '''
        This function will compute the decryption key given the encryption key and phi
        :param e:
        :param phi:
        :return:
        '''
        # since there exists a case such that the difference between phi and e are ginormous, if we iterate d
        # starting from 1 it is too computationally expensive and since we can pick any d such that the following
        # function is True then we don;t have to be so picky about the starting points.
        d = 1
        while True:
            if d * e % phi == 1:
                return d
            d += 1


    def decrypt(self):
        return pow(self.cipher_text, self.d, self.public_key[1])
