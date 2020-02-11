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

<<<<<<< HEAD
def f(x):
    return np.matmul([[0, -1], [1, 0]], x)

dt = .05
=======
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
>>>>>>> 7a1b7aed0f80bfa112e5b8465f6940fe3394566d
for dt in [dt, 10/dt]:
    time = []
    trajectory = []
    t = 0
    x = np.array([1, 0])
    for i in range(int(10/dt)):
        time.append(t)
        trajectory.append(x)
        t = t + dt
<<<<<<< HEAD
        x = x + f(x)*dt

    plt.plot(time, trajectory)

=======
        x = x + f(0, x)*dt
        # x = rk4_update(x)
        # x = fancyMethod.y
    plt.plot(times, trajectory)
    # plt.plot(fancyMethod.t, np.transpose(fancyMethod.y))
>>>>>>> 7a1b7aed0f80bfa112e5b8465f6940fe3394566d
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.tight_layout()
plt.show()
