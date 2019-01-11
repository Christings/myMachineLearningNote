import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0]  # (97,)
y = data[:, 1]

fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax1.scatter(X, y, marker='x')
# plt.show()

m = len(y)
X = np.reshape(X, (m, 1))  # (97,1)
X = np.hstack((np.ones((m, 1)), X))  # 给X增加一行,(97,2)
y = np.reshape(y, (m, 1))
theta = np.zeros((2, 1))


def computeCost(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    J = 1 / (2 * m) * sum((h - y) ** 2)
    # print(J[0])
    return J[0]


computeCost(X, y, theta)
computeCost(X, y, np.array([[2], [1]]))

iterations = 1500
alpha = 0.01


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        h = np.dot(X, theta)
        theta = theta - alpha * np.dot(np.transpose(X), (h - y)) / m
        J_history[i] = computeCost(X, y, theta)
        print(J_history[i])

    return theta, J_history


theta, J_history = gradientDescent(X, y, theta, alpha, iterations)


def plotJ(J_history, iterations):
    x = np.arange(1, iterations + 1)
    ax2 = fig1.add_subplot(222)
    ax2.plot(x, J_history)
    ax2.set_xlabel('iterations')
    # plt.xlabel('iterations')
    ax2.set_ylabel('loss')
    ax2.set_title('iterations vs loss')
    # ax2.set_title()
    # plt.show()


plotJ(J_history, iterations)


def plot_result(X, y, theta):
    ax3 = fig1.add_subplot(223)
    ax3.scatter(X[:, 1], y)
    # ax3.hold(True) # 已废弃，不加这个参数也可以叠加
    # ax3.cla() # 加上这个可以清除上一个图
    ax3.plot(X[:, 1], np.asarray(np.dot(X, theta)), color='r')
    plt.show()


plot_result(X, y, theta)
