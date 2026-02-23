import numpy as np

class MultivariateNormal:
    """
    Multivariate normal N(mu, Sigma) with stable Cholesky-based logpdf/pdf and sampling.

    Parameters
    ----------
    mu : array_like, shape (D,)
        Mean vector.
    Sigma : array_like, shape (D, D)
        Covariance matrix (must be symmetric positive definite).
    rng : np.random.Generator, optional
        Random generator for rvs(). If None, uses default_rng().

    Notes
    -----
    - logpdf(x) accepts x with shape (D,) or (N, D) and returns scalar or (N,).
    - rvs(shape) returns samples with shape (*shape, D) (SciPy-like).
    """
    def __init__(self, mu, Sigma, rng=None):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)

        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array of shape (D,).")
        self.D = self.mu.shape[0]

        if self.Sigma.shape != (self.D, self.D):
            raise ValueError(f"Sigma must have shape ({self.D}, {self.D}).")

        # Cholesky factor: Sigma = L L^T (requires PD covariance)
        self.L = np.linalg.cholesky(self.Sigma)

        # Precompute constants for logpdf
        self.log_det = 2.0 * np.sum(np.log(np.diag(self.L)))
        self.log_norm = -0.5 * (self.D * np.log(2.0 * np.pi) + self.log_det)

        self.rng = rng if rng is not None else np.random.default_rng()

    def logpdf(self, x):
        """
        Compute log density at x.

        Parameters
        ----------
        x : array_like, shape (D,) or (N, D)

        Returns
        -------
        logp : float or np.ndarray
            Scalar if x is (D,), else shape (N,).
        """
        x = np.asarray(x, dtype=float)
        x2d = x.reshape(1, -1) if x.ndim == 1 else x

        if x2d.ndim != 2 or x2d.shape[1] != self.D:
            raise ValueError(f"x must have shape (D,) or (N, D) with D={self.D}.")

        diff = x2d - self.mu                      # (N, D)
        y = np.linalg.solve(self.L, diff.T)       # (D, N)
        quad = np.sum(y * y, axis=0)              # (N,)

        out = self.log_norm - 0.5 * quad
        return float(out[0]) if x.ndim == 1 else out

    def pdf(self, x):
        """Compute density at x (may underflow in high D)."""
        return np.exp(self.logpdf(x))

    def rvs(self, shape=()):
        """
        Draw random samples.

        Parameters
        ----------
        shape : int or tuple, optional
            Number of samples or sample shape. If shape=k (int), returns (k, D).
            If shape=(a,b), returns (a, b, D). Default is () => returns (D,).

        Returns
        -------
        samples : np.ndarray
            Samples with shape (*shape, D).
        """
        if isinstance(shape, int):
            shape = (shape,)

        z = self.rng.standard_normal(size=(*shape, self.D))  # (*shape, D)
        return self.mu + z @ self.L.T