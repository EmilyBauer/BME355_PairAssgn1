"""
BME 355 - Assignment 1 Q1
BME355_1.py
Sarah Schwartzel    20710946
Emily Bauer         20727725
2020/02/09
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.matmul([[0, -1], [1, 0]], x)

dt = .05
for dt in [dt, 10/dt]:
    time = []
    trajectory = []
    t = 0
    x = np.array([1, 0])
    for i in range(int(10/dt)):
        time.append(t)
        trajectory.append(x)
        t = t + dt
        x = x + f(x)*dt

    plt.plot(time, trajectory)

plt.xlabel('Time (s)')
plt.ylabel('State')
plt.tight_layout()
plt.show()


def rk4_update(x):
    s1 = f(t, x)
    s2 = f(t + dt/2, x + dt/2*s1)
    s3 = f(t + dt/2, x + dt/2*s2)
    s4 = f(t + dt, x + dt*s2)
    x = x + dt/6*(s1 + 2*s2 + 2*s3 + s4)


