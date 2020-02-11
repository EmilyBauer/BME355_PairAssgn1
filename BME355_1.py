"""
BME 355 - Assignment 1 Q1
BME355_1.py
Sarah Schwartzel    20710946
Emily Bauer         20727725
2020/02/09
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(t, x):
    return np.matmul([[0, -1], [1, 0]], x)
def rk4_update(x):
    
        s1 = f(t, x)
        s2 = f(t + dt/2, x + dt/2*s1)
        s3 = f(t + dt/2, x + dt/2*s2)
        s4 = f(t + dt, x + dt*s2)
        x = x + dt/6*(s1 + 2*s2 + 2*s3 + s4)
        return x

fancyMethod = solve_ivp(f, [0, 10], [1,0])
dt = 0.25
for dt in [dt, 10/dt]:
    time = []
    trajectory = []
    t = 0
    x = np.array([1, 0])
    for i in range(int(10/dt)):
        time.append(t)
        trajectory.append(x)
        t = t + dt
        x = x + f(0, x)*dt
        # x = rk4_update(x)
        # x = fancyMethod.y
    plt.plot(times, trajectory)
    # plt.plot(fancyMethod.t, np.transpose(fancyMethod.y))
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.tight_layout()
plt.show()
