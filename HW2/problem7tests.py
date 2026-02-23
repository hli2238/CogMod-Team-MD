import numpy as np
from problem7 import multivariate_normal_density

def run_scipy_comparisons():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)

    tests = []

    # --- Spherical Gaussian ---
    D = 3
    mu = np.zeros(D)
    sigma2 = 2.0
    Sigma = sigma2 * np.eye(D)
    x = rng.normal(size=(5, D))
    tests.append(("spherical", x, mu, Sigma))

    # --- Diagonal Gaussian ---
    D = 4
    mu = np.array([1.0, -2.0, 0.5, 3.0])
    variances = np.array([0.5, 2.0, 1.5, 4.0])
    Sigma = np.diag(variances)
    x = rng.normal(size=(5, D))
    tests.append(("diagonal", x, mu, Sigma))

    # --- Full covariance Gaussian ---
    D = 3
    mu = np.array([0.2, -0.1, 1.0])
    A = rng.normal(size=(D, D))
    Sigma = A @ A.T + 0.2 * np.eye(D)  # make PD
    x = rng.normal(size=(5, D))
    tests.append(("full", x, mu, Sigma))

    for name, x, mu, Sigma in tests:
        my_pdf = multivariate_normal_density(x, mu, Sigma)
        sp_pdf = multivariate_normal(mean=mu, cov=Sigma).pdf(x)

        max_abs = np.max(np.abs(my_pdf - sp_pdf))
        max_rel = np.max(np.abs(my_pdf - sp_pdf) / (np.abs(sp_pdf) + 1e-300))

        print(f"{name:10s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")

if __name__ == "__main__":
    run_scipy_comparisons()