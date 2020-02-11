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

a,b = 1000,1        #force-velocity curve const
alpha = 20000       #spring const (N/m)
p0 = 2000           #isometric force (N)
g= 9.807            #gravity
m=220               #mass (kg)

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

dt = 0.0078125
time = np.arange(0,2,dt)
x = np.array([0,0,0])
x_trajectory = []
for t in time:
    # x = x + dt*f(x)
    xa = f(x)
    xb = f(x+dt*xa)
    x = x+1/2*dt*(xb+xa)
    x_trajectory.append(x)




x_trajectory = np.array(x_trajectory)
plt.subplot(3,1,1)
plt.plot(time, x_trajectory[:,0])
plt.ylabel('Tension (N)')
plt.subplot(3,1,2)
plt.plot(time, x_trajectory[:,1])
plt.ylabel('Velocity (m/s)')
plt.subplot(3,1,3)
plt.plot(time, x_trajectory[:,2])
plt.ylabel('Position (m)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('Q2_0.0078125dt.png')
plt.show()

#Trapezoidal method
# smthn smth smthn

