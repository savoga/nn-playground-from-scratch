import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

class NeuralNetwork():

    def __init__(self, learning_rate=0.1):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.y_iterations = []

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
        self.weights = np.random.normal(0, 1, (n_features, 1))
        self.bias = np.random.normal(0, 1)
        # self.weights = np.zeros((n_features, 1))
        # self.bias = 0
        # self.weights = np.random.uniform(2, 5, (n_features, 1))
        # self.bias = np.random.uniform(2, 5)

        for i in range(n_iterations):

            print('---------- iteration {} ----------'.format(i+1))

            # Forward propagation
            z = np.dot(self.weights.T, X_train) + self.bias # /!!\ z is of dimension (1, n_samples)
            a = self.sigmoid(z)
            cost = self.cost(a, y_train)
            print('cost: {}'.format(cost))
            # For tracking purposes
            y_iter_proba = a
            y_iter = np.array([1 if proba>0.5 else 0 for proba in y_iter_proba.flatten()])
            self.y_iterations.append(y_iter)

            # Backward propagation
            dw = (1/n_samples)*(np.dot((a - y_train), X_train.T))
            db = (1/n_samples)*np.sum(a - y_train)

            # Update parameters
            self.weights = self.weights - self.learning_rate*dw.T
            self.bias = self.bias - self.learning_rate*db

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
X_train, y_train = make_moons(n_samples=100,
                              random_state=3)
plt.scatter(X_train[:,0], X_train[:,1])

n_iterations = 10
nn = NeuralNetwork(learning_rate=0.5)
y_iterations = nn.train(X_train, y_train, n_iterations)

for i in range(n_iterations):
    plt.figure()
    y_iter = y_iterations[i]
    X_positive = X_train[np.where(y_iter==1)]
    plt.scatter(X_positive[:,0], X_positive[:,1], color='red')
    X_negative = X_train[np.where(y_iter==0)]
    plt.scatter(X_negative[:,0], X_negative[:,1], color='blue')
    plt.title('{} iteration'.format(i))
    plt.show()

# X_test, y_test = make_blobs(n_samples=100, n_features=2, centers=2)
# y_pred_proba = nn.predict(X_test)
# y_pred = [1 if proba>0.5 else 0 for proba in y_pred_proba.flatten()]



'''
- Double check why it's not a mess before the first iteration
- Try generating moons
- Find a difficult case
- Add layer 2
'''