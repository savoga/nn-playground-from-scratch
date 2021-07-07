import numpy as np

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

        np.random.seed(seed=2)
        self.weights_1 = np.random.randn(self.nodes_hidden, n_features)*0.01
        self.bias_1 = 0 #np.random.normal(0, 1, (1, 1))
        self.weights_2 = np.random.randn(1, self.nodes_hidden)*0.01
        self.bias_2 = 0 #np.random.normal(0, 1, (1, 1))

        for i in range(n_iterations):

            # Forward propagation
            a1 = np.tanh(np.dot(self.weights_1, X_train) + self.bias_1)
            a2 = self.sigmoid(np.dot(self.weights_2, a1) + self.bias_2)
            cost = self.cost(a2, y_train)
            if i % 100 == 0:
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
            dz1 = np.multiply(np.dot(self.weights_2.T,dz2),1-np.power(a1,2))
            dw1 = (1/n_samples)*np.dot(dz1,X_train.T)
            db1 = (1/n_samples)*np.sum(dz1, axis = 1, keepdims = True)

            # Update parameters
            self.weights_1 = self.weights_1 - self.learning_rate*dw1
            self.bias_1 = self.bias_1 - self.learning_rate*db1
            self.weights_2 = self.weights_2 - self.learning_rate*dw2
            self.bias_2 = self.bias_2 - self.learning_rate*db2

        return self.y_iterations

    def predict(self, X_test):
        X_test = X_test.T # (n_feature, n_samples)
        return self.sigmoid(np.dot(self.weights.T, X_test) + self.bias)