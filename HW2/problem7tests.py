import numpy as np
from scipy.stats import multivariate_normal

from problem7 import MultivariateNormal 


def run_all_tests():
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
    Sigma = A @ A.T + 0.2 * np.eye(D)  # positive definite
    x = rng.normal(size=(5, D))
    tests.append(("full", x, mu, Sigma))

    for name, x_batch, mu, Sigma in tests:
        sp = multivariate_normal(mean=mu, cov=Sigma)
        my = MultivariateNormal(mu, Sigma)  # if your constructor differs, adjust here

        # ---------- PDF comparison (batch) ----------
        my_pdf = my.pdf(x_batch)
        sp_pdf = sp.pdf(x_batch)

        max_abs_pdf = np.max(np.abs(my_pdf - sp_pdf))
        max_rel_pdf = np.max(np.abs(my_pdf - sp_pdf) / (np.abs(sp_pdf) + 1e-300))

        # ---------- LOGPDF comparison (batch) ----------
        my_logpdf = my.logpdf(x_batch)
        sp_logpdf = sp.logpdf(x_batch)

        max_abs_log = np.max(np.abs(my_logpdf - sp_logpdf))
        max_rel_log = np.max(np.abs(my_logpdf - sp_logpdf) / (np.abs(sp_logpdf) + 1e-300))

        # ---------- Single x comparison ----------
        x_single = x_batch[0]
        my_pdf_1 = my.pdf(x_single)
        sp_pdf_1 = sp.pdf(x_single)
        my_log_1 = my.logpdf(x_single)
        sp_log_1 = sp.logpdf(x_single)

        abs_pdf_1 = abs(my_pdf_1 - sp_pdf_1)
        abs_log_1 = abs(my_log_1 - sp_log_1)

        # ---------- rvs() checks ----------
        n = 200_000
        samples = my.rvs(n)  # should be (n, D)
        shape_ok = samples.shape == (n, mu.shape[0])

        sample_mean = samples.mean(axis=0)
        sample_cov = np.cov(samples, rowvar=False)

        mean_err = np.max(np.abs(sample_mean - mu))
        cov_err = np.max(np.abs(sample_cov - Sigma))

        print(f"\n{name.upper()}")
        print(f"pdf batch:    max_abs={max_abs_pdf:.3e}  max_rel={max_rel_pdf:.3e}")
        print(f"logpdf batch: max_abs={max_abs_log:.3e}  max_rel={max_rel_log:.3e}")
        print(f"pdf single:   abs_err={abs_pdf_1:.3e}")
        print(f"logpdf single:abs_err={abs_log_1:.3e}")
        print(f"rvs shape ok? {shape_ok}  (got {samples.shape}, expected {(n, mu.shape[0])})")
        print(f"rvs mean max|err|={mean_err:.3e}")
        print(f"rvs cov  max|err|={cov_err:.3e}")

        # Optional: quick pass/fail thresholds (you can tweak)
        if max_abs_pdf > 1e-10 or max_abs_log > 1e-10:
            print("WARNING: pdf/logpdf mismatch seems larger than expected.")
        if not shape_ok:
            print("WARNING: rvs() returned wrong shape.")


if __name__ == "__main__":
    run_all_tests()