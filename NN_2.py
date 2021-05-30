import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

class NeuralNetwork():

    def __init__(self, learning_rate=0.1, nodes_hidden=2):
        self.weights_1 = None # weights first layer
        self.bias_1 = None # bias first layer
        self.weights_2 = None # weights second layer
        self.bias_2 = None # bias second layer
        self.learning_rate = learning_rate
        self.y_iterations = []
        self.nodes_hidden = nodes_hidden

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y_hat, y):
        return ((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))

    def cost(self, y_hat, y):
        m = y_hat.shape[1] # number of training examples
        loss = self.loss(y_hat, y)
        return -(1/m)*np.sum(loss)


    def train(self, X_train, y_train, n_iterations):

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        X_train = X_train.T # (n_features, n_samples)
        y_train = y_train.T # (n_features, 1)

        np.random.seed(seed=1)
        self.weights_1 = np.random.normal(0, 1, (self.nodes_hidden, n_features))
        self.bias_1 = np.random.normal(0, 1, (1, 1))
        self.weights_2 = np.random.normal(0, 1, (1, self.nodes_hidden))
        self.bias_2 = np.random.normal(0, 1, (1, 1))

        for i in range(n_iterations):

            # Forward propagation
            a1 = self.sigmoid(np.dot(self.weights_1, X_train) + self.bias_1)
            a2 = self.sigmoid(np.dot(self.weights_2, a1) + self.bias_2)
            cost = self.cost(a2, y_train)
            if i % 10000 == 0:
                print('---------- iteration {} ----------'.format(i+1))
                print('cost: {}'.format(cost))
            # For tracking purposes
            y_iter_proba = a2
            y_iter = np.array([1 if proba>0.5 else 0 for proba in y_iter_proba.flatten()])
            self.y_iterations.append(y_iter)

            # Backward propagation
            dz2 = a2-y_train
            dw2 = (1/n_samples)*np.dot(dz2,a1.T)
            db2 = (1/n_samples)*np.sum(dz2)
            dz1 = np.multiply(np.dot(self.weights_2.T,dz2),np.multiply(a1,1-a1))
            dw1 = (1/n_samples)*np.dot(dz1,X_train.T)
            db1 = (1/n_samples)*np.sum(a1-y_train)

            # Update parameters
            self.weights_2 = self.weights_2 - self.learning_rate*dw2
            self.bias_2 = self.bias_2 - self.learning_rate*db2
            self.weights_1 = self.weights_1 - self.learning_rate*dw1
            self.bias_1 = self.bias_1 - self.learning_rate*db1

        return self.y_iterations

    def predict(self, X_test):
        X_test = X_test.T # (n_feature, n_samples)
        return self.sigmoid(np.dot(self.weights.T, X_test) + self.bias)

# X_train, y_train = make_blobs(n_samples=100,
#                               n_features=2,
#                               centers=2,
#                               cluster_std=0.3,
#                               center_box=(0,5),
#                               random_state=3)
# X_train, y_train = make_circles(n_samples=100,
#                               random_state=3)
X_train, y_train = make_moons(n_samples=500,
                              random_state=3,
                              noise=.05)
plt.scatter(X_train[:,0], X_train[:,1])
plt.show()

# Good visual
# n_samples=500, noise=.05, n_iterations=170K, learning_rate=0.75, nodes_hidden=5
# n_samples = 500, noise=.2, n_iterations=170K, learning_rate=1.2, nodes_hidden=5

n_iterations = 170000
nn = NeuralNetwork(learning_rate=0.75, nodes_hidden=5)
y_iterations = nn.train(X_train, y_train, n_iterations)

# X_test, y_test = make_blobs(n_samples=100, n_features=2, centers=2)
# y_pred_proba = nn.predict(X_test)
# y_pred = [1 if proba>0.5 else 0 for proba in y_pred_proba.flatten()]

for i in [n_iterations-1]:#range(n_iterations):
    if True:# i%10000==0:
        plt.figure()
        y_iter = y_iterations[i]
        X_positive = X_train[np.where(y_iter==1)]
        plt.scatter(X_positive[:,0], X_positive[:,1], color='red')
        X_negative = X_train[np.where(y_iter==0)]
        plt.scatter(X_negative[:,0], X_negative[:,1], color='blue')
        plt.title('{} iteration'.format(i))
        plt.show()

'''
Site:
    difference one layer and two layers (input layer doesn't count)
    image for 2 layers and 2 hidden nodes
'''