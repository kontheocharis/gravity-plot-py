import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

G = 6.67e-11 # in SI units
no_of_masses = 4
dim = 2

# time is in seconds
time = np.arange(0, 3.154e+7 * 4, 86400. / 4)

# positions
r = np.zeros((len(time), no_of_masses, dim))

# velocities
v = np.zeros((len(time), no_of_masses, dim))

# masses (in kg)
m = np.array([
    4.87e24, # venus
    5.97e24, # earth
    7.35e22, # moon
    1.99e30, # sun
])

# Initial positions [x, y](t=0)
r[0] = [
    [108.7e9, 0.],  # venus
    [151.9e9, 0.],  # earth
    [384.4e6 + 151.9e9, 0.],  # moon
    [0., 0.],  # sun
]

# Initial velocities [dx/dt, dy/dt](t=0)
v[0] = [
    [0., 35000.],  # venus
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

# plot
fig = plt.figure()
ax3d = fig.add_subplot(2, 1, 1, projection='3d')
ax2d = fig.add_subplot(2, 1, 2)

relative_to_i = 3
for i in range(no_of_masses):
    ax2d.scatter(r[:, i, 0] - r[:, relative_to_i, 0], r[:, i, 1] - r[:, relative_to_i, 1], c=time)
    ax3d.scatter(r[:, i, 0] - r[:, relative_to_i, 0], r[:, i, 1] - r[:, relative_to_i, 1], time, c=time)


plt.show()
# plt.savefig('plot1.png')
