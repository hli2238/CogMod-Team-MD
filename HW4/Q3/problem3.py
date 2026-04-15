"""
HW4 Problem 3: Speed Dating logistic regression in Stan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



HW4 = Path(__file__).resolve().parent
STAN_FILE = HW4 / "speed_dating_logistic.stan"
LOCAL_CSV = HW4 / "speed_dating_data.csv"

KAGGLE_SLUG = "annavictoria/speed-dating-experiment"
KAGGLE_MAIN_CSV = "Speed Dating Data.csv"

# Data key 
ATTR_TO_COL = {
    "attractiveness": "attr",
    "sincerity": "sinc",
    "intelligence": "intel",
    "fun": "fun",
    "ambition": "amb",
    "shared_interests": "shar",
}

# chosen attributes (attractiveness, intelligence, fun)
CHOSEN = ("attractiveness", "intelligence", "fun")

TEST_FRACTION = 0.2
RNG_SEED = 4210


def step1_load_dataframe() -> pd.DataFrame:
    """Load the Speed Dating Experiment CSV from disk or Kaggle.

    If :data:`LOCAL_CSV` exists, it is read directly. Otherwise the dataset is
    fetched via ``kagglehub`` (with a filesystem fallback if the pandas adapter
    fails).

    Returns
    -------
    pd.DataFrame
        Full raw table; encoding ``ISO-8859-1``, ``low_memory=False``.

    Raises
    ------
    RuntimeError
        If the Kaggle download path contains no ``*.csv`` files.
    """
    if LOCAL_CSV.is_file():
        return pd.read_csv(LOCAL_CSV, encoding="ISO-8859-1", low_memory=False)

    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    try:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_SLUG,
            KAGGLE_MAIN_CSV,
            pandas_kwargs={"encoding": "ISO-8859-1", "low_memory": False},
        )
    except Exception as e:
        root = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        csvs = sorted(root.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if not csvs:
            raise RuntimeError(f"No CSV under {root}") from e
        return pd.read_csv(csvs[0], encoding="ISO-8859-1", low_memory=False)


def step2_extract_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build binary outcome and predictor matrix from the raw table.

    Uses :data:`CHOSEN` and :data:`ATTR_TO_COL` to select columns. Coerces
    values to numeric, drops rows with any missing outcome or predictor, and
    exits the process if required columns are absent.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Speed Dating data (must include ``dec`` and predictor columns).

    Returns
    -------
    X : np.ndarray
        Shape ``(n_samples, 3)``, float64 predictor matrix.
    y : np.ndarray
        Shape ``(n_samples,)``, integer binary outcome (``dec``).
    cols : list of str
        Short column names used for ``X`` (e.g. ``attr``, ``intel``, ``fun``).

    Notes
    -----
    Calls :func:`sys.exit` if ``dec`` or a required predictor column is missing.
    """
    if "dec" not in df.columns:
        sys.exit(f"Expected column 'dec'. First columns: {list(df.columns)[:25]}")

    cols = [ATTR_TO_COL[name] for name in CHOSEN]
    for c in cols:
        if c not in df.columns:
            sys.exit(f"Missing column {c!r}. Check ATTR_TO_COL / CHOSEN.")

    use = df[["dec"] + cols].copy()
    use["dec"] = pd.to_numeric(use["dec"], errors="coerce")
    for c in cols:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna()
    y = use["dec"].astype(int).to_numpy()
    X = use[cols].to_numpy(dtype=np.float64)
    return X, y, cols


def step3_train_test_and_standardize(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split and z-score predictors using training stats only.

    Training mean and standard deviation are applied to both train and test
    sets; zero-variance columns get scale 1.0 to avoid division by zero.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix, shape ``(n, k)``.
    y : np.ndarray
        Binary labels, shape ``(n,)``.

    Returns
    -------
    X_tr_s : np.ndarray
        Standardized training predictors.
    X_te_s : np.ndarray
        Standardized test predictors (using training ``mu``, ``sd``).
    y_tr : np.ndarray
        Training labels.
    y_te : np.ndarray
        Test labels.
    mu : np.ndarray
        Per-column training means, shape ``(k,)``.
    sd : np.ndarray
        Per-column training standard deviations (population ``ddof=0``), shape ``(k,)``.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RNG_SEED, stratify=y
    )
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd
    return X_tr_s, X_te_s, y_tr, y_te, mu, sd


def step4_fit_stan(
    X_tr_s: np.ndarray,
    X_te_s: np.ndarray,
    y_tr: np.ndarray,
) -> "cmdstanpy.CmdStanMC":
    """Compile :data:`STAN_FILE` and sample from the posterior.

    Passes training rows for likelihood and test rows as ``x_test`` for
    generated quantities (e.g. ``p_hat``), if defined in the Stan program.

    Parameters
    ----------
    X_tr_s : np.ndarray
        Standardized training design matrix.
    X_te_s : np.ndarray
        Standardized test design matrix (same number of columns as train).
    y_tr : np.ndarray
        Training binary outcomes.

    Returns
    -------
    cmdstanpy.CmdStanMC
        Fitted object after four chains, 1000 warmup and 1000 sampling iterations.
    """
    import cmdstanpy

    N, K = X_tr_s.shape
    M = X_te_s.shape[0]
    stan_data = {
        "N": N,
        "K": K,
        "x": X_tr_s,
        "y": y_tr.astype(int).tolist(),
        "M": M,
        "x_test": X_te_s,
    }

    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        show_progress=True,
        show_console=False,
    )
    return fit


def step5_summarize(fit: "cmdstanpy.CmdStanMC") -> None:
    """Print compact posterior summaries for intercept and coefficients.

    Shows means and intervals for ``alpha`` and ``beta[1]``…``beta[K]``, plus
    global diagnostics (max R-hat, min bulk ESS) and which predictor has the
    largest absolute posterior mean coefficient on the standardized scale.

    Parameters
    ----------
    fit : cmdstanpy.CmdStanMC
        Object returned by :func:`step4_fit_stan`.

    Notes
    -----
    Writes to stdout only; does not return a value.
    """
    summ = fit.summary(sig_figs=4)
    rows = ["alpha"] + [f"beta[{i + 1}]" for i in range(len(CHOSEN))]
    sub = summ.loc[[r for r in rows if r in summ.index]]

    want_cols = ["Mean", "2.5%", "97.5%", "5%", "95%", "ESS_bulk", "ESS_tail", "R_hat"]
    out_cols = [c for c in want_cols if c in sub.columns]
    print(sub[out_cols])
    for i, name in enumerate(CHOSEN):
        print(f"  beta[{i + 1}] = {name}")

    max_rhat = float(summ["R_hat"].max())
    min_ess_bulk = float(summ["ESS_bulk"].min())
    means = sub["Mean"].to_numpy()
    beta_means = means[1 : 1 + len(CHOSEN)]
    strongest = CHOSEN[int(np.argmax(np.abs(beta_means)))]
    print(
        f"max R_hat (all params)={max_rhat:.4f}  min ESS_bulk={min_ess_bulk:.0f}  "
        f"largest |E[beta]| on std scale: {strongest}"
    )


def main() -> None:
    df = step1_load_dataframe()
    X, y, _cols = step2_extract_xy(df)
    X_tr_s, X_te_s, y_tr, y_te, _mu, _sd = step3_train_test_and_standardize(X, y)

    fit = step4_fit_stan(X_tr_s, X_te_s, y_tr)
    step5_summarize(fit)

    try:
        p_hat = fit.stan_variable("p_hat")
        out = HW4 / "problem3_fit_meta.npz"
        np.savez_compressed(out, y_test=y_te, p_hat_samples=p_hat, predictors=np.array(CHOSEN))
    except Exception as e:
        print(f"Could not save {HW4 / 'problem3_fit_meta.npz'}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
