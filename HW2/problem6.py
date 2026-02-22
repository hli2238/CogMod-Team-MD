import numpy as np
import matplotlib.pyplot as plt

def posterior(p, s, c):
    return (s * p) / (s * p + (1 - c) * (1 - p))

# Fixed baseline values
prior = 0.01
sensitivity = 0.95
specificity = 0.90

# Posterior vs prior graph with fixed sensitivity and specificity
p_vals = np.linspace(0.001, 0.5, 500)
post_prior = posterior(p_vals, sensitivity, specificity)

plt.figure(figsize=(6,4))
plt.plot(p_vals, post_prior)
plt.title("Posterior vs Prior")
plt.xlabel("Prior Probability of Disease")
plt.ylabel("Posterior P(D|+)")
plt.grid()
plt.show()

# Posterior vs sensitivity graph with fixed prior and specificity
s_vals = np.linspace(0.01, 1, 500)
post_sens = posterior(prior, s_vals, specificity)

plt.figure(figsize=(6,4))
plt.plot(s_vals, post_sens)
plt.title("Posterior vs Sensitivity")
plt.xlabel("Sensitivity")
plt.ylabel("Posterior P(D|+)")
plt.grid()
plt.show()

# Posterior vs specificity graph with fixed prior and sensitivity
c_vals = np.linspace(0.01, 1, 500)
post_spec = posterior(prior, sensitivity, c_vals)

plt.figure(figsize=(6,4))
plt.plot(c_vals, post_spec)
plt.title("Posterior vs Specificity")
plt.xlabel("Specificity")
plt.ylabel("Posterior P(D|+)")
plt.grid()
plt.show()