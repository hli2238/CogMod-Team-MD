import numpy as np

def multivariate_normal_density(x, mu, Sigma):
    """
    Density of a D-dim multivariate normal N(mu, Sigma) evaluated at x.

    Parameters
    ----------
    x : array_like, shape (D,) or (N, D)
    mu : array_like, shape (D,)
    Sigma : array_like, shape (D, D)

    Returns
    -------
    pdf : float or np.ndarray
        If x is (D,), returns scalar. If x is (N, D), returns shape (N,).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array of shape (D,).")
    D = mu.shape[0]
    if Sigma.shape != (D, D):
        raise ValueError(f"Sigma must have shape ({D},{D}).")

    x2d = x.reshape(1, -1) if x.ndim == 1 else x
    if x2d.shape[1] != D:
        raise ValueError(f"x must have last dimension D={D}.")

    L = np.linalg.cholesky(Sigma)
    diff = (x2d - mu)

    y = np.linalg.solve(L, diff.T)              # shape (D, N)
    quad = np.sum(y * y, axis=0)                # shape (N,)

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    log_norm = -0.5 * (D * np.log(2.0 * np.pi) + log_det)

    log_pdf = log_norm - 0.5 * quad
    pdf = np.exp(log_pdf)

    return float(pdf[0]) if x.ndim == 1 else pdf