import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def posterior(p, s, c):
    return (s * p) / (s * p + (1 - c) * (1 - p))

# Create grids
grid_points = 200

p_vals = np.linspace(0.001, 0.5, grid_points)
s_vals = np.linspace(0.01, 1, grid_points)
c_vals = np.linspace(0.01, 1, grid_points)

# Posterior vs prior graph with fixed sensitivity and specificity
P, S = np.meshgrid(p_vals, s_vals) # use meshgrid to create 2D grids
C_fixed = 0.90
Z1 = posterior(P, S, C_fixed)

# Substituted plt.plot with plt.figure and ax.plot_surface for 3D surface plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P, S, Z1, cmap='viridis')
ax.set_title("Posterior vs Prior & Sensitivity")
ax.set_xlabel("Prior")
ax.set_ylabel("Sensitivity")
ax.set_zlabel("Posterior")

# Replaced plt.show to save the figure instead
plt.savefig("graph1.png")

# Posterior vs specificity graph with fixed prior and sensitivity
P, C = np.meshgrid(p_vals, c_vals)
S_fixed = 0.95
Z2 = posterior(P, S_fixed, C)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P, C, Z2, cmap='plasma')
ax.set_title("Posterior vs Prior & Specificity")
ax.set_xlabel("Prior")
ax.set_ylabel("Specificity")
ax.set_zlabel("Posterior")
plt.savefig("graph2.png")

# Posterior vs sensitivity & specificity graph with fixed prior
S, C = np.meshgrid(s_vals, c_vals)
P_fixed = 0.01
Z3 = posterior(P_fixed, S, C)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, C, Z3, cmap='inferno')
ax.set_title("Posterior vs Sensitivity & Specificity")
ax.set_xlabel("Sensitivity")
ax.set_ylabel("Specificity")
ax.set_zlabel("Posterior")
plt.savefig("graph3.png")