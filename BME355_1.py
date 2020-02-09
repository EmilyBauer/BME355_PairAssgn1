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
a,b = 1000,1
alpha = 20000
p0 = 2000
g= 9.807
m=220


def f(x):
    """
    :param x: state vector
    :return: derivative of state vector wrt time
    """
    return np.array([
        alpha * (x[1] + b*(p0-x[0])/(x[0]+a)),
        g - x[0]/m,
        x[1]
    ])

initState = [1, 0]

dt = 0.05
time = np.arange(0,10,dt)
x = np.array([0,0,0])
xVal = []
for t in time:
    x = x+dt*f(x)
    xVal.append(x)

xVal = np.array(xVal)
plt.subplot(3,1,1)
plt.plot(time, xVal[:,0])
plt.ylabel('Tension (N)')
plt.subplot(3,1,2)
plt.plot(time, xVal[:,1])
plt.ylabel('Velocity (m/s)')
plt.subplot(3,1,3)
plt.plot(time, xVal[:,2])
plt.ylabel('Position (m)')
plt.xlabel('Time (s)')
plt.tight_layout()
# plt.savefig('hill-mass-simulation.png')
plt.show()

# # dt = 0.001
# x = np.array([0, 0, 0]) #initial condition
# for i in range(100):
#     x = x+dt*f(x)   #Euler step

# # print(x)


# def getFeatures(state):
def rk4_update(x):
    s1 = f(t, x)
    s2 = f(t + dt/2, x + dt/2*s1)
    s3 = f(t + dt/2, x + dt/2*s2)
    s4 = f(t + dt, x + dt*s2)
    x = x + dt/6*(s1 + 2*s2 + 2*s3 + s4)


