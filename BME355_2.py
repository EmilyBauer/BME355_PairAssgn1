"""
BME 355 - Assignment 1 Q2
BME355_2.py
Sarah Schwartzel    20710946
Emily Bauer         20727725
2020/02/09
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

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

dt = 0.1
time = np.arange(0,5,dt)
x = np.array([0,0,0])

xVal = []
for t in time:
    if t == 0:
        x = initState + dt*f(x)
        xVal.append(x)
        print(x)
    else:
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
plt.savefig('hill-mass-simulation.png')