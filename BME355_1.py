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

def f(x):
    """
    :param x: state vector
    :return: derivative of state vector wrt time
    """
    return np.dot(A, x)

initState = np.array([[1,0]])
dt = 0.05
duration = 10
time = np.arange(0,duration,dt)
x = np.array([[0],[0]])
xVal = []
for t in time:
    if t == 0:
        x = initState + dt*f(x)
        xVal.append(x)
    else:
        x = x+dt*f(x)
        xVal.append(x)

xVal = np.array(xVal)
plt.plot(time, xVal[:,0])
plt.ylabel('State')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()


def rk4_update(x):
    s1 = f(t, x)
    s2 = f(t + dt/2, x + dt/2*s1)
    s3 = f(t + dt/2, x + dt/2*s2)
    s4 = f(t + dt, x + dt*s2)
    x = x + dt/6*(s1 + 2*s2 + 2*s3 + s4)


