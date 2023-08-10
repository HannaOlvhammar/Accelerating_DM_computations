import numpy as np
import matplotlib.pyplot as plt
import struct


data_size = 1000
n_Q = 7


#params = np.fromfile('../parameters.dat')
params = np.loadtxt('../cuda/parameters.dat')
#rates = np.fromfile('../rates_for_different_Qs.dat')
rates = np.loadtxt('../cuda/rates.dat')
print(np.shape(params))
print(np.shape(rates))
rates = np.reshape(rates, (data_size,n_Q))

X = np.load('params.npy')
Y = np.load('Qs.npy')
print(X)
err = 1e-6

for i in range(data_size):
    for j in range(n_Q):
        y_py = Y[i,j]
        y_c = rates[i,j]
        print(str(y_py) + ' AND ' + str(y_c))
        assert (y_c <= y_py + err*y_py) and (y_c >= y_py - err*y_py)


for i in range(data_size):
    for j in range(3):
        x_py = X[i,j]
        x_c = params[i,j]
        print(str(x_py) + ' AND ' + str(x_c))
        assert (x_c <= x_py + err*x_py) and (x_c >= x_py - err*x_py)


print()
print("Everything is perfect!")
print()
