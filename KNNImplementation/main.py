'''
This file is a simple demonstration of K-Nearest-Neighbour algorithm, the file will proceed in the following steps
1. read data from csv file
2. Normalize the data
3. spilt the data in 2 groups, training set and verifying set
4. calculate the distance of each element in verifying set to training set and allocate the index for the top k num
   nearest neighbour in the training set
5. use the index from the previous steps to evaluate corresponding y value
6. find the majority y value among the k num of y value in the each element in the verifying set and return that value
7. use the list we obtain from the previous step and compare with the actual y value in verifying set and eventually form
   a report such that conclude the following: accuracy, marco average, weighted average
'''
import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv("diabetes.csv")
# reading data from csv file
Y = df['y'].values
X = df.drop(['y'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=104, test_size=0.20, shuffle=True)
# split testing and training set in 2 group with 2 : 8 ratio
scaler = StandardScaler()
# Standard Scaler Normalization
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def Cos_distance(X_train_arr_label, X_test_arr_label):
    '''
    This function is for calculating the distance between the training set and the verifying set by using
    the cosine distance function
    :return:
    '''
    neumerator = 0
    X_train_denominator = 0
    X_test_denominator = 0
    for i in range(0, 7):
        # The following implementation is based on the cosin similarity formula
        neumerator += (X_train[X_train_arr_label][i] * X_test[X_test_arr_label][i])
        # print(type(X_train[X_train_arr_label][i]))
        X_train_denominator += X_train[X_train_arr_label][i] ** 2
        X_test_denominator += X_test[X_test_arr_label][i] ** 2
    denominator = sqrt(X_train_denominator) * sqrt(X_test_denominator)
    # Note: cosin distance = 1 - cosin similarity
    return 1 - neumerator / denominator


def Calc_KNN_perLabel(X_test_arr_label, k):
    '''
    This function is to calculate and store the cousin distance for each training set and sort the distance in ascending
    order In the dictionary, the key would be the distance, the value would be the index of the distance. Thus we will have
    the top 3 (since we picked our k value to be 3) closest distance in the beginning of the dictionary along with
    its value
    :param X_test_arr_label:
    :param k:
    :return:
    '''
    Distance_list_perLabel = {}
    KNN_index_List = []
    for i in range(0, len(X_train)):
        # store the index of the distance in Distance_list_perlabel dictionary with their distance be the key
        Distance_list_perLabel[Cos_distance(i, X_test_arr_label)] = i
    distance_list = sorted(Distance_list_perLabel)  # sort the distance(key) in ascending order
    for j in range(0, k):
        # store the top k num of distances' index in KNN_list
        KNN_index_List.append(Distance_list_perLabel[distance_list[j]])
    return KNN_index_List


def Calc_Distance_list():
    '''
    This function we will calculate the KNN index for each element in verifying set and store them in a list, thus eventually
    we will have a nested list such that each sub-list contains 3 nearest neighbour for each verifying set elements.
    :return:
    '''
    Distance_list = []
    for i in range(0, len(X_test)):
        Distance_list.append(Calc_KNN_perLabel(i, 3))
    return Distance_list


def Calc_y(index_of_KNN):
    '''
    This function we will match the corresponding y value based on the previous value for x nearest neighbour and store them
    in a nested list, thus the sub-list in the main list will contain the corresponding y value
    :param index_of_KNN:
    :return:
    '''
    y_train_list_temp = []
    for i in range(0, len(index_of_KNN)):
        for j in range(0, 3):
            y_train_list_temp.append(y_train[index_of_KNN[i][j]])  # store 3 index of the y value from y train
        y_train_list.append(y_train_list_temp)
        y_train_list_temp = y_train_list_temp * 0  # clearing the y_train_list temp


def return_Majority(y_train_arr):
    '''
    This function comapre and return the majority in the sub-list in y_train_list, by doing so, we can predict the nearest
    neighbour with the highest matches y value in the y train set
    :param y_train_arr:
    :return:
    '''
    counter_for_one = 0
    counter_for_zero = 0
    majority_y_test = []
    for i in range(0, len(y_train_arr)):
        for j in range(0, 3):
            if y_train_arr[i][j] == 0:
                counter_for_zero += 1
            else:
                counter_for_one += 1
        if counter_for_one > counter_for_zero:
            majority_y_test.append(1)
        else:
            majority_y_test.append(0)
        counter_for_zero = 0
        counter_for_one = 0
    return majority_y_test


y_train_list = []
Calc_y(Calc_Distance_list())

# From here we are generating the classification report
y_pred = return_Majority(y_train_list)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
