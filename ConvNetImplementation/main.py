'''

'''
# The below libraries are used in the assignment
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==================================================================#
# Load the mnist dataset using keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Filter data for digits 1, 2, 3 based on the assignemnt requirement
train_filter = np.where((train_labels == 1) | (
        train_labels == 2) | (train_labels == 3))
test_filter = np.where((test_labels == 1) | (
        test_labels == 2) | (test_labels == 3))
train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]

X = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

# Flatten and normalize the images
train_images = train_images.reshape(train_images.shape[0], 28 * 28) / 255
test_images = test_images.reshape(test_images.shape[0], 28 * 28) / 255


# ==================================================================#
# Convert labels to one-hot vectors for example class 1 one hot vector is [1,0,0]
def one_hot_encode(labels):
    '''
    This function takes an array containing the labels and convert into one hot vector, below is all the potential one hot
    vectors given the project is classifying 1, 2 and 3
    1 -> [1, 0, 0]
    2 -> [0, 1, 0]
    3 -> [0, 0, 1]
    :param labels:
    :return:
    '''
    one_hot = []
    for i in range(len(labels)):
        sub_arr = np.zeros(3)  # create a np array size 3 and filled with 0
        sub_arr[labels[i] - 1] = 1  # modify the sub_arr index of label value - 1 to be 1, the reason we subtract 1 is
        # np array index starts from 0
        one_hot.append(sub_arr)
    return np.array(one_hot)


train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)
# ==================================================================#

# Initialize weights and bias
weights = np.random.randn(784, 3)
bias = np.random.randn(1, 3)

# ==================================================================#

'''
The following function is our softmax function. 
'''


def softmax(x):
    xExp = np.exp(x - np.max(x))
    probabilities = xExp / xExp.sum()
    return probabilities


'''
The following function is the cross entrophy function.
It is used to penilize miss classification.

y_true is the label
y_pred is the predicted
'''


# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    loss = 0
    for i in range(len(y_pred)):
        loss += (y_true[i] * np.log(1e-8 + y_pred[i]))
    return -sum(loss) / len(y_pred)


# ==================================================================#

'''
The following function is used to find the weights and bias for our model.
It follows the steps outlined in the assingment instructions.
'''


def train(images, labels, num_epochs, lr):
    # Training the network
    bias = np.random.randn(1, 3)
    weights = np.random.randn(784, 3)

    # Setting a list for loss per epoch
    loss_list = []

    for epoch in range(num_epochs):
        # Forward pass
        logits = np.dot(images, weights) + bias
        predictions = softmax(logits)

        # Compute loss using cross entropy
        loss = cross_entropy_loss(labels, predictions)

        loss_list.append(loss)

        print("Epoch:", epoch, "Loss:", loss)

        # Backward pass
        # Gradient of loss with respect to logits
        dl_dz = predictions - labels

        # Gradient of loss with respect to the weights
        n = images.shape[0]

        dl_dw = (1 / n) * images.T.dot(dl_dz)

        # Gradient of loss with respect to the bias
        dl_db = (1 / n) * np.sum(dl_dz, axis=0)

        # Update weights and biases
        weights = weights - (lr * dl_dw)

        bias = bias - (lr * dl_db)

    return weights, bias, np.mean(loss_list)


# getting weights and bias to be used with the def accuracy
weightsRegAccuracy, biasRegAccuracy, avg_loss = train(train_images, train_labels, 100, 0.2)
print()

# ==================================================================#

# Evaluate the model
'''
Function takes an array contianing y_true values, 
and an array containing y_pred values. Returns a number
that represents the accuracy of the model.
'''


def accuracy(y_true, y_pred):
    # return the accuracy
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    correct = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == y_true[i]):
            correct += 1
    return correct / len(y_pred)


# make a predictions on the test set
logits = np.dot(test_images, weightsRegAccuracy) + biasRegAccuracy
# Apply softmax
test_predictions = softmax(logits)
print("Test accuracy:", accuracy(test_labels, test_predictions) * 100, "%")
print("\n")
# #==================================================================

'''
The following function is our cross validation function. x are the input features, y are the labels.
This function is used to evaluate the performence of the model.
'''


def cross_validate(X, y, n_folds=5):
    scores = []

    print("Cross Validation begins ...")
    print("\n")

    for i in range(n_folds):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / n_folds, random_state=0)

        y_train_one_hot = one_hot_encode(y_train)
        y_val_one_hot = one_hot_encode(y_val)

        weights, bias, avg_loss = train(X_train.reshape(X_train.shape[0], 28 * 28) / 255, y_train_one_hot, 50, 0.4)

        print("\n")

        # make predictions on the validation set
        logits = np.dot(X_val.reshape(X_val.shape[0], 28 * 28) / 255, weights) + bias

        # Apply softmax
        val_predictions = softmax(logits)

        # Calculate accuracy
        acc = accuracy_score(np.argmax(y_val_one_hot, axis=1), np.argmax(val_predictions, axis=1))
        print(acc)
        scores.append(acc)
    return np.mean(scores)


print("")
print(f"Cross-validation accuracy: {cross_validate(X, y) * 100:.2f}%")
print("")
print("")

# Ploting the accuracy as a function of learning rate
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracy_lr = []

for lr in learning_rates:
    weights, bias, avg_loss = train(train_images, train_labels, 50, lr)  # Training using the training set
    # Getting logits from testing set
    logits = np.dot(test_images, weights) + bias

    # Applying softmax
    val_predictions = softmax(logits)

    # Calculating accuracy
    acc = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(val_predictions, axis=1))
    accuracy_lr.append(acc)

plt.figure(figsize=(8, 6))
plt.plot(learning_rates, accuracy_lr, marker='o')
plt.title('Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plot the accuracy as a function of # of epochs

accuracy_ep = []
loss_list = []
epochs = [i for i in range(1, 101)]
for epoch_num in epochs:
    weights, bias, avg_loss = train(train_images, train_labels, epoch_num, 0.4)
    # Getting logits from testing set
    logits = np.dot(test_images, weights) + bias

    # Applying softmaxx
    val_predictions = softmax(logits)

    # Calculating accuracy
    acc = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(val_predictions, axis=1))
    accuracy_ep.append(acc)

    # Adding loss to the loss list

    loss_list.append(avg_loss)

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_ep, marker='o')
plt.title('Accuracy vs. Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plot the loss as a function of # of epochs
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_list, marker='o')
plt.title('Average Loss per epoch vs. Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
