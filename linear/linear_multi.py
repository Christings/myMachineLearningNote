import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2:3]


def feature_normalization(X):
    X_norm = X

    column_mean = np.mean(X_norm, axis=0) #
    print('mean=', column_mean)
    column_std = np.std(X_norm, axis=0)
    print('std=', column_std)

    X_norm = X_norm - column_mean
    X_norm = X_norm / column_std

    return column_mean, column_std, X_norm


means, stds, X_norm = feature_normalization(X)

m = len(y)
X_norm = np.hstack((np.ones((m, 1)), X_norm))  # (47,3)

theta = np.zeros((X_norm.shape[1], 1))


def computeCost(X, y, theta):
    m = len(y)
    # J=0
    h = np.dot(X, theta)
    J = 1 / (2 * m) * sum((h - y) ** 2)
    return J[0]


computeCost(X_norm, y, theta)

alpha = 0.01
iterations = 400


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        h = np.dot(X, theta)
        theta = theta - alpha * np.dot(np.transpose(X), (h - y)) / m
        J_history[i] = computeCost(X, y, theta)
        print(J_history[i])
    return theta, J_history


theta, J_history = gradientDescent(X_norm, y, theta, alpha, iterations)


def plotJ(J_history, iterations):
    x = np.arange(1, iterations + 1)
    plt.plot(x, J_history)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('iterations of loss')
    plt.show()


plotJ(J_history, iterations)
