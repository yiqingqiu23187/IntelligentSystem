import numpy as np
import matplotlib.pyplot as plt
import random
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

x = np.linspace(-np.pi, np.pi, 10)
y = np.sin(x)
x_test = np.random.random(10) * np.pi / 2
y_test = np.sin(x_test)

hide = 10  # 设置隐藏层神经元个数,可以改着玩
W1 = np.random.random((hide, 1))
B1 = np.random.random((hide, 1))
W2 = np.random.random((1, hide))
B2 = np.random.random((1, 1))
learningrate = 0.06
iteration = 5000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


E = np.zeros((iteration, 1))
Y = np.zeros((10, 1))

for k in range(iteration):
    temp = 0
    for i in range(10):
        hide_in = np.dot(x[i], W1) - B1
        hide_out = sigmoid(hide_in)
        y_out = np.dot(W2, hide_out) - B2
        Y[i] = y_out
        e = y_out - y[i]
        dB2 = -1 * learningrate * e
        dW2 = e * learningrate * np.transpose(hide_out)
        dB1 = np.zeros((hide, 1))
        for j in range(hide):
            dB1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * (-1) * e * learningrate)
        dW1 = np.zeros((hide, 1))
        for j in range(hide):
            dW1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * x[i] * e * learningrate)

        W1 = W1 - dW1
        B1 = B1 - dB1
        W2 = W2 - dW2
        B2 = B2 - dB2
        temp = temp + abs(e)

    E[k] = temp

test = np.arange(-np.pi, np.pi, 0.1)
result = np.zeros(len(test))
for i in range(len(test)):
    hide_in = np.dot(test[i], W1) - B1
    hide_out = sigmoid(hide_in)
    result[i] = np.dot(W2, hide_out) - B2


plt.plot(test, result, label='myresult')
plt.scatter(test, result, label='myresult')
plt.plot(test, np.sin(test), c='r', label='sin(x)')
plt.title('sin')
plt.legend()
plt.show()

plt.scatter(x, y, c='r')
plt.plot(x, Y)
plt.title('test set:red point   test result:blue line')
plt.show()

plt.scatter(x_test, y_test, c='y', marker='+')
plt.plot(x, Y)
plt.title('test set:red point   test result:blue line')
plt.show()

xx = np.linspace(0, iteration, iteration)
plt.plot(xx, E)
plt.title('lost function')
plt.show()
