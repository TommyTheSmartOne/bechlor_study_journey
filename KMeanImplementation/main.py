'''
This file is an implementation of K-Mean algorithm, the attached csv file is our dataset, K-Mean is use for data clustering
in a data set, K is the amount of centroid(group) the data have, we picked it. The algorithm will be proceed in the following
steps(Notice I have 2 step 4 because they both "work" but 4* out perform 4 in many cases, more detailed explanation see
documentation on randomly_select_data_point_as_centroids()):
1. read data from csv file
2. Normalize the data
3. identify the domain for the centroids(don't want the centroids to be to far from data points, optimize the time complexity)
4. Randomly generate the centroids, depends on what k value we picked for the given data set
4*. Randomly generate the initial centroids from the given datasets based on the k value
5. calculate the corresponding cos distance from each centroids to each data points
6. grouping the data points with the closer centroids
7. upload the centroids position thus closer to its group
8. upload the group since step 7 will change the location of the centroids, the distance of each data points to centroids will
   also change.
9. repeat step 7 and 8 until we find the optimal solution, that is, iteration n+1 = iteration n(the position of centroid)
   is no longer changing
10. generate the report based on a separate data set to test the accuracy rate
'''
import csv
import random
import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


df = pd.read_csv("zero_and_one.csv")
# reading data from csv file

scaler = StandardScaler()
# Standard Scaler Normalization
scaler.fit(df)
Data_after_normalize = scaler.transform(df)


def identify_the_domain_of_one_parameter(data, label_serial):
    '''
    This function is dedicate to allocate the greatest value within one feature of the dataset, make sure the input for
    label_serial must be numerical variable since categorical data will not be taken into consideration when implementing
    .
    :param data:
    :param label_serial:
    :return:
    '''
    collectionOfParameters = []
    for i in range(0, len(data)):
        collectionOfParameters.append(data[i][label_serial])
    sort_collection_of_parameter = sorted(collectionOfParameters)
    return [sort_collection_of_parameter[len(collectionOfParameters) - 1], sort_collection_of_parameter[0]]


def identify_the_domain(data):
    '''
    This function is for applying the domain for every feature given
    :param data:
    :return:
    '''
    Max_Min_range_of_each_feature = []
    for i in range(0, len(data[0])):
        Max_Min_range_of_each_feature.append(identify_the_domain_of_one_parameter(data, i))
    return Max_Min_range_of_each_feature


def randomly_generate_centroid(data, k):
    '''
    This dunction is for randomly generate the initial centroid value
    :param data:
    :param k:
    :return:
    '''
    Max_Min_range_of_each_feature = identify_the_domain(data)
    float_Max_Min_range_of_each_feature = np.array(Max_Min_range_of_each_feature).tolist()
    Centroid_list = []
    temp_list = []
    for i in range(0, k):
        for j in range(0, len(float_Max_Min_range_of_each_feature)):
            temp_list.append(random.uniform(float_Max_Min_range_of_each_feature[j][1],
                                            float_Max_Min_range_of_each_feature[j][0]))
        Centroid_list.append(temp_list)
        temp_list = temp_list * 0
    return Centroid_list


def randomly_select_data_point_as_centroids():
    '''
    This function is for pick a data points randomly from the given data sets, it differentiates from randomly_generate_centroids()
    since that function is randomly pick a point from the cartesian plane from the given domain(based on the MAX and MIN
    value of the dataset). Which is inaccurate due to the randomness, if say we generate 2 centroids(k = 2) denoted as
    a and b, the distance between centroid a to every other points in the given dataset is longer than centroid b to any
    other data point, then centroid a will not be assigned with any data points in its group and centroid b will obtain all
    the data points. Thus the algorithm will eventually be problematic and inaccurate
    :return:
    '''
    centroid_list = []
    for i in range(0, k):
        index_of_centroid = random.randint(0, len(df))
        centroid_list.append(Data_after_normalize[index_of_centroid])
    return centroid_list


def cos_distance(Centroid_list, normalized_label, centroid_label):
    '''
    This function is for calculating the cosin distance between the centroid and the data points by using
    the cosine distance function
    :return:
    '''
    numerator = 0
    centroid_denominator = 0
    data_point_denominator = 0
    for i in range(0, len(Data_after_normalize[0])):
        numerator += Data_after_normalize[normalized_label][i] * Centroid_list[centroid_label][i]
        data_point_denominator += Data_after_normalize[normalized_label][i] ** 2
        centroid_denominator += Centroid_list[centroid_label][i] ** 2
    denominator = sqrt(data_point_denominator) * sqrt(centroid_denominator)

    # Note: cosin distance = 1 - cosin similarity
    return 1 - numerator / denominator


def calc_cos_distance_for_each_data_point(Centroid_list, Normalized_data_list):
    '''
    This function is for calculate the distance between each data point and each centroids, we will return a list with len
    n such that n is the num of centroid we picked(k value).
    :param Centroid_list:
    :param Normalized_data_list:
    :return:
    '''
    distance_list = []
    temp_list = []
    for i in range(0, len(Centroid_list)):
        for j in range(0, len(Normalized_data_list)):
            temp_list.append(cos_distance(Centroid_list, j, i))
        distance_list.append(temp_list)
        temp_list = temp_list * 0
    return distance_list


def grouping(distance_list):
    '''
    This function is for group the data points along with its centroids
    :param distance_list:
    :return:
    '''
    groups = []
    temp_dict = {}
    for j in range(k):
        groups.append([])
    for i in range(0, len(distance_list[0])):
        for m in range(0, len(distance_list)):
            temp_dict[distance_list[m][i]] = m
        sort_dict = sorted(temp_dict)
        Shortest_centroid_index = temp_dict[sort_dict[0]]
        groups[Shortest_centroid_index].append(i)
        temp_dict.clear()
    return groups


def upload_centroid(grouped_list):
    '''
    This function is for uploading the centroids to the mean of each value in the cluster
    :return:
    '''
    temp_list = []
    mean_list_per_centroids = []
    for i in range(0, len(grouped_list)):
        for j in range(0, len(grouped_list[i])):
            temp_list.append(Data_after_normalize[grouped_list[i][j]])
        mean_list_per_centroids.append(calc_mean(temp_list))
        temp_list = temp_list * 0
    return mean_list_per_centroids


def calc_mean(temp_list):
    '''
    This function is for calculate the mean of each column
    :param temp_list:
    :return:
    '''
    counter = 0
    mean = []
    while counter < len(temp_list[0]):
        mean_for_each_feature = 0
        for i in range(0, len(temp_list)):
            mean_for_each_feature += temp_list[i][counter]
        mean_for_each_feature = mean_for_each_feature / len(temp_list)
        mean.append(mean_for_each_feature)
        counter += 1
    return mean


def optimize_centroids(iteration):
    '''
    Repeat the process of grouping, calc mean and upload_centroids until n+1 th iteration == n th iteration or the destinatied
    iteration num is reached
    :param iteration:
    :return:
    '''
    pbar = tqdm(total=1000)
    counter = 0
    # centroids_list = Randomly_generate_centroid(Data_after_normalize, k)
    centroids_list = randomly_select_data_point_as_centroids()
    grouped_list = grouping(calc_cos_distance_for_each_data_point(centroids_list, Data_after_normalize))
    uploaded_centroids = upload_centroid(grouped_list)
    while is_centroids_still_changing(centroids_list, uploaded_centroids) and counter < iteration:
        centroids_list = uploaded_centroids
        grouped_list = grouping(calc_cos_distance_for_each_data_point(centroids_list, Data_after_normalize))
        uploaded_centroids = upload_centroid(grouped_list)
        counter += 1
        pbar.update(1)
    pbar.close()
    return grouped_list


def is_centroids_still_changing(centroid, centroid_after_one_iteration):
    '''
    Check if the centroids are still changing
    :param centroid:
    :param centroid_after_one_iteration:
    :return:
    '''
    return not np.all(np.equal(centroid, centroid_after_one_iteration))


k = 2
optimized_groups = optimize_centroids(1000)
print(optimized_groups)  # obtain the results
