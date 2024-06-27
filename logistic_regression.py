import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression
from sklearn.datasets import load_diabetes


PRECISION = 10 ** (-8)


def logistic_regression_linear(X, y):
    """
        Question: how to present p(X) = Pr(Y = 1|X)?
        Here I set p(X) = 1 if Y = 1 given X and p(X) = 0 otherwise

        return:
          coefficient: np.array, coefficients in logistic function for constant
            part and features in X.
    """

    pr = np.empty((len(y), 1))
    # for the sake of log, pr cannot be exact one or zero
    pr[y > 0] = 1 - PRECISION # very close to 1
    pr[y == 0] = PRECISION # very close to 0

    logit = np.log(pr / (1 - pr))
    coefficient = linear_regression(X, logit)

    return coefficient


class LogisticRegression:
    def __init__(self, iteration=1000, learning_rate=0.001, error_rate=0.001):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.error_rate = error_rate
        self.errors = []
        self.coef = None

    @staticmethod
    def sigmoid(X, coef):
        y_sigmoid = 1 / (1 + np.exp(-X @ coef))
        return y_sigmoid

    def fit(self, X, y, add_intercept=True):
        if add_intercept: # add constant at the first column of X
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        coef = np.zeros((X.shape[1], 1))
        for _ in range(iteration):
            y_sigmoid = self.sigmoid(X, coef)

            y_hat = np.where(y_sigmoid >= 0.5, 1, 0)
            error = np.sum(y != y_hat) / len(y)
            self.errors.append(error)
            if error <= self.error_rate: # acceptable error
                break

            # renew coefficients
            coef += self.learning_rate * (X.T @ (y - y_sigmoid))

        self.coef = coef

    def predict(self, x, add_intercept=True):
        if add_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        y_hat = self.sigmoid(x, self.coef)
        y_hat = np.where(y_hat >= 0.5, 1, 0)
        return y_hat


class LogisticRegressionMultinomial:
    def __init__(self, iteration=1000, learning_rate=0.001, error_rate=0.001):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.error_rate = error_rate
        self.errors = []
        self.coef = None

    @staticmethod
    def sigmoid(X, coef):
        y_hat = np.exp(X @ coef)
        sum_y_hat = np.sum(y_hat, axis=1).reshape(-1, 1)
        y_hat = np.divide(y_hat, sum_y_hat)
        return y_hat

    def fit(self, X, y, add_intercept=True):
        # y needs to be integer array and starts from 0 to K-1 one by one.
        # y needs to be flat.

        if add_intercept: # add constant at the first column of X
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        K = len(set(y)) # numbers of classes in y
        coefs = np.zeros((X.shape[1], K))
        ones = np.zeros((X.shape[0], K)) # dummies
        for i in range(K):
            ones[y == i, i] = 1

        for _ in range(self.iteration):
            y_hat = self.sigmoid(X, coefs)
            
            error = np.sum(np.argmax(y_hat, axis=1) != y) / len(y)
            self.errors.append(error)
            if error <= self.error_rate: # meet maximum error criterion
                break

            coefs += self.learning_rate * X.T @ (ones - y_hat)

        self.coef = coefs

    def predict(self, x, add_intercept=True):
        if add_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        y_hat = self.sigmoid(x, self.coef)
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat


if __name__ == '__main__':
    datapoints = 1000
    features = 1
    classes = 4
    random_radius = 0.2
    iteration = 4000 
    learning_rate = 0.01
    error_rate = 0.01

    # data for y classes > 2
    X = np.random.rand(datapoints, features) * 10
    y = np.zeros(datapoints)

    x = np.sum(X, axis=1)
    indices = x.argsort()
    interval = int(datapoints / classes)
    for i in range(1, classes):
        y[x >= x[indices[interval * i]]] = i

    # for classes == 2
#    radius = int(datapoints * random_radius)
#    mid = int(datapoints / 2)
#    X = np.linspace(0, 10, datapoints).reshape(-1, 1)
#    XX = np.hstack((np.ones((X.shape[0], 1)), X))
#    y = np.zeros(datapoints)
#    y[mid:] = 1
##    y = np.ones(datapoints)
##    y[mid:] = 0
#    np.random.shuffle(y[mid-radius: mid+radius])

    # multinomial logistic regression
    l = LogisticRegressionMultinomial(iteration, learning_rate, error_rate)
    l.fit(X, y)
    y_hat = l.predict(X)
    error = np.sum(y_hat != y) / len(y)

    print(f'final error: {l.errors[-1]}, error: {error}')
    plt.plot(range(len(l.errors)), l.errors)

    # binary logistic regression: fit and predict (maximum likelihood)
#    l = LogisticRegression(iteration, learning_rate, error_rate)
#    l.fit(X, y.reshape(-1, 1))
#    y_hat = l.predict(X)
#    error = np.sum(y_hat != y.reshape(-1, 1)) / len(y)
#    print(f'error (ML): {l.errors[-1]}, error: {error}')

#    plt.figure('old')
#    plt.plot(range(len(l.errors)), l.errors)

    # fit and predict (linear)
#    coef = logistic_regression_linear(X, y)
#    y_linear = np.where(l.sigmoid(XX, coef) >= 0.5, 1, 0)
#    print(f'error (linear): {np.sum(y != y_linear) / len(y)}, coef: {coef.T}')

    # plot
#    sigmoid_ML = l.sigmoid(XX, l.coef)
#    sigmoid_linear = l.sigmoid(XX, coef)
#    plt.plot(X, sigmoid_ML)
#    plt.plot(X, sigmoid_linear, 'g')
#    plt.scatter(X, y, c='r', s=0.1)

    plt.show()
