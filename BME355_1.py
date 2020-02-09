"""
BME 355 - Assignment 1
BME355_1.py
Sarah Schwartzel    20710946
Emily Bauer         20727725
2020/02/09
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

A = np.array([[0,-1],[1,0]])
w, v = la.eig(A)


print(w)
# # print(v)


# time = np.array()

# state = idek    #init position

dt = 0.05
time = np.arange(0,10,dt)
x = np.array([0,0,0])
xVal = []
for t in time:
    x += dt*f(x)
    xVal.append(x)

xVal = np.array(xVal)
plt.subplot(3,1,1)


# def f(x):
#     """
#     :param x: state vector
#     :return: derivative of state vector wrt time
#     """
#     return np.array([
#         alpha * (x[1] + b*(p0-x[0])/(x[0]+a)),
#         g - x[0]/m
#         x[1]
#     ])
# # dt = 0.001
# x = np.array([0, 0, 0]) #initial condition
# for i in range(100):
#     x = x+dt*f(x)   #Euler step

# # print(x)


# def getFeatures(state):




