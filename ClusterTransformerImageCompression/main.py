'''
This file is an idea of myself using a transforming function to transfer the data to resolve data cluster in a more
efficient way.
NOTE: the file is just an idea i derive and it does not guarantee to work
The project will proceed in the following way:
1. transfer a input image into a 3d matrix
2. apply transform math function to find out the clustering(grouping), the transformer function will be describe down below
3. normalize the 3d matrix to make its value between 0 and 1
4. use dimension reduction technique to reduce the dimension to 2 in this case, the input would be the RGB color and output
    would be the data point index in our matrix
5. input the RGB color (dimension reduced) into the math function we discovered, the output of the function will help
    us determine the 'centroid' the data point belongs to
Transformer function:
    As mentioned before, the transformer function must be symmetric and its y value must be approaching 2 value such that
    f(x) <= y
    In this case we can choose the sigmoid function: f(x) = 1/(1 + e^(-x))
    The sigmoid function will guarantee the functions output between 0 and 1 and since it is 'symmetric' with the middle
    point 0.5. In our case, the if the output y value is greater than 0.5, the data point belongs to the group with 1 as
    centroid, if the output y is smaller than 0.5, we can consider the value belongs to the group such that centroid is
    0.
'''
import numpy as np
import cv2 as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_data(image_address):
    '''
    This function read a image and convert it to a 3d array, each 2d array contains each row of pixel in the image, each
    1d array within the 2d array contains data related to its RGB data
    :return:
    '''
    img = cv2.imread(image_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Scaling the image so that the values are in the range of 0 to 1
    img = img / 255
    return img


# We now obtained a 3d array with RBG colors corresponding to each pixel
image_arr = preprocess_data("bird.png")
