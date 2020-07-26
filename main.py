#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


G = 6.67e-11 # in SI units
no_of_masses = 4
dim = 2

# time is in seconds
time = np.arange(0, 3.154e+7 * 4, 86400. / 4)

# masses (in kg)
m = np.array([
    4.87e10, # venus
    5.97e20, # earth
    7.35e22, # moon
    1.99e30, # sun
])


def calculate(deviation):

    # positions
    r = np.zeros((len(time), no_of_masses, dim))

    # velocities
    v = np.zeros((len(time), no_of_masses, dim))

    # Initial positions [x, y](t=0)
    r[0] = [
        [108.7e9 + deviation, 0.],  # venus
        [151.9e9, 0.],  # earth
        [384.4e6 + 151.9e9, 0.],  # moon
        [0., 0.],  # sun
    ]

    # Initial velocities [dx/dt, dy/dt](t=0)
    v[0] = [
        [0., 10000.],  # venus
        [0., 30000.],  # earth
        [0., 30000. + 1000.],  # moon
        [0., 0.],  # sun
    ]


    # calculate the acceleration due to gravity from m at a displacement s
    def a_gravity(m, s):
        s_mag2 = np.dot(s, s)
        if s_mag2 == 0:
            s_mag2 = 1
        return (G * m / s_mag2) * (s / np.sqrt(s_mag2))

    # simulate
    for t in range(1, len(time)):
        dt = time[t] - time[t - 1]

        a = np.zeros((no_of_masses, dim))

        for i in range(no_of_masses):
            for j in range(no_of_masses):
                if i == j:
                    continue
                a[i] += a_gravity(m[j], r[t - 1][j] - r[t - 1][i])

        v[t] = v[t - 1] + a * dt
        r[t] = r[t - 1] + v[t] * dt

    return r, v

r, v = calculate(deviation=0)
r_d, v_d = calculate(deviation=1000)

sep = r_d[:][0] - r[:][0]
sep_l = np.log(np.linalg.norm(r_d[:, 0] - r[:, 0], axis=1) / np.linalg.norm(r_d[0, 0] - r[0, 0]))

# fig = plt.figure()
# ax2d = fig.add_subplot(2, 1, 2)

# ax2d.scatter(time, sep_l)

time_range = slice(0, 1087)
slope, _, _, _, _ = stats.linregress(time[time_range], sep_l[time_range])
print(slope / 1087)

# sep_l = np.linalg.norm(sep, axis=0)

# for t in range(len(time)):
#     print(sep_l[t])

# # plot

# ax2d2 = fig.add_subplot(2, 1, 1)

# relative_to_i = 3
# for i in range(no_of_masses):
#     ax2d2.scatter(r[:, i, 0] - r[:, relative_to_i, 0], r[:, i, 1] - r[:, relative_to_i, 1], c=time)


# plt.show()
# # # plt.savefig('plot1.png')
