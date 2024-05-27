'''
This file contains an implementation of using K-Mean algorithm to compress image size by reducing the num of color in a
given image. The implementation will proceed in the following steps:
1. Transfer image to a matrix size n x m where n is the width and m is the height of the matrix.
2. randomly select k num of data points as centroids where k is define by us, in this case we select k as 16
3. group data points to a centroid based on the euclidean distance between the centroids and the data points, the data
    points will be assign to the closest centroids
4. shift centroids to a new data points where this data points' euclidean distance has the least differences with the mean
    distance in that group compare to other data points
5. repeat step 3 and 4 until one of the following condition meet:
    a). num of epoch reaches desire epoch number
    b). the centroids stopped shifting
6. replace every pixels RGB color with its corresponding centroids RGB color and plot the image using matplotlib.pyplot
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





def kMean_ini_centroids(imageInput, num_of_centroid):
    '''
    This function randomly selects data points as initial centroids depending on the number of centroids we pick for our
    data set
    :param imageInput:
    :param num_of_centroid:
    :return:
    '''
    ini_centroids_list = []
    for i in range(num_of_centroid):
        row_index = np.random.randint(0, len(imageInput))
        col_index = np.random.randint(0, len(imageInput[0]))
        ini_centroids_list.append(imageInput[row_index][col_index])

    return ini_centroids_list


def euclidean_distance(centroid_list, row_num, col_num):
    '''
    This function is for calculating the euclidean distance between the centroid and the data points by using
    the euclidean distance function
    :return:
    '''
    euclidean_distance_dict = {}
    for j in range(len(centroid_list)):
        sum_sq = 0
        for i in range(len(centroid_list[j])):
            sum_sq += (centroid_list[j][i] - image_arr[row_num][col_num][i]) ** 2
        euclidean_distance_dict[np.sqrt(sum_sq)] = j
    smallest_index = sorted(euclidean_distance_dict)[0]
    return smallest_index, euclidean_distance_dict[smallest_index]


def find_closest_centroid(centroid_list):
    '''
    This function is for calculate the distance between each data point and each centroids, we will return a list with len
    n such that n is the num pixel image contains.
    :param centroid_list:
    :return: closest_centroid_dict: this dictionary contains index of centroid list as key and distances for all pixel
            belongs to that index group as value
    '''
    centroid_and_distance_dict = {}
    centroid_and_index_dict = {}
    for k in range(0, len(centroid_list)):  # initialize 2 dictionary where the first one centroid_and_distance_dict:
        # with key : centroid_index, value : distance from data points to that centroid --> list, the second one
        # centroid_index_dict: with key : centroid_index, value : index of data point in the image_arr --> 2d list
        centroid_and_distance_dict.update({k: []})
        centroid_and_index_dict.update({k: []})
    for i in range(0, len(image_arr)):
        for j in range(0, len(image_arr[i])):
            distance, centroid_index = euclidean_distance(centroid_list, i, j)
            centroid_and_distance_dict[centroid_index].append(distance)
            centroid_and_index_dict[centroid_index].append([i, j])
    return centroid_and_distance_dict, centroid_and_index_dict


def compute_centroids(centroid_list, centroid_index_pair):
    '''
    This function compute the new centroids based on the group list we obtained from previous function, we will first sum
    up the total in one group, then devide by the number of data point belongs to that specific centroid thus obtain the
    mean distance in a centroid group. We now have k num of mean distance(k is the num of centroids we pick). Finally we
    will call the locate_closest_datapoint_to_new_centroid()
    :param centroid_list:
    :param centroid_distance_pair:
    :param centroid_index_pair:
    :return:
    '''
    group_mean_list = []
    for i in range(len(centroid_list)):
        R_sum = 0
        G_sum = 0
        B_sum = 0
        for j in range (len(centroid_index_pair[i])):
            R_sum += image_arr[centroid_index_pair[i][j][0]][centroid_index_pair[i][j][1]][0]
            G_sum += image_arr[centroid_index_pair[i][j][0]][centroid_index_pair[i][j][1]][1]
            B_sum += image_arr[centroid_index_pair[i][j][0]][centroid_index_pair[i][j][1]][2]
        R_sum /= len(centroid_index_pair[i])
        G_sum /= len(centroid_index_pair[i])
        B_sum /= len(centroid_index_pair[i])
        group_mean_list.append([R_sum, G_sum, B_sum])
    return group_mean_list





def if_centroid_shift(previous_centroids_list, current_centroids_list):
    '''
    This function checks if the centroids is still shifting between previous iteration and current iteration
    :param previous_centroids_list:
    :param current_centroids_list:
    :return:
    '''
    return not np.all(np.equal(previous_centroids_list, current_centroids_list))


def run_kMean():
    '''
    This function is the main loop of k-mean algorithm.
    :return:
    '''
    pbar = tqdm(total=NUM_OF_ITERATION)  # progress bar
    counter = 0
    previous_centroid_list = kMean_ini_centroids(image_arr, NUM_OF_CENTROID)
    centroid_distance_pair, centroid_index_pair = find_closest_centroid(previous_centroid_list)
    current_centroid_list = compute_centroids(previous_centroid_list , centroid_index_pair)
    while if_centroid_shift(previous_centroid_list, current_centroid_list) and counter < NUM_OF_ITERATION:
        previous_centroid_list = current_centroid_list
        centroid_distance_pair, centroid_index_pair = find_closest_centroid(previous_centroid_list)
        current_centroid_list = compute_centroids(previous_centroid_list, centroid_index_pair)
        counter += 1
        pbar.update(1)
    pbar.close()
    return centroid_index_pair, current_centroid_list


def compress_image():
    '''
    This function takes the processed matrix and transfer it to an image where the num of RGB color = k
    :return:
    '''
    for i in range(len(final_centroid_index_pair)):
        # replace all the pixel color with its corresponding centroid color
        for j in range(len(final_centroid_index_pair[i])):
            image_arr[final_centroid_index_pair[i][j][0]][final_centroid_index_pair[i][j][1]] = final_centroid[i]

    # plotting the compressed image.
    plt.imshow(image_arr)
    plt.show()
    return


#  Constant initialization and function call
image_arr = preprocess_data("bird.png")
NUM_OF_ITERATION = 2
NUM_OF_CENTROID = 10
final_centroid_index_pair, final_centroid = run_kMean()
compress_image()
