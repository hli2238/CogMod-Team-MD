"""
Cognitive Modeling HW4 — Problem 3: load data, split, standardize, fit logistic regression in Stan.

Prints a compact posterior summary (for write-up). Saves problem3_fit_meta.npz for Problem 4.

Dependencies (use a venv if needed):
  pip install cmdstanpy pandas numpy scikit-learn 'kagglehub[pandas-datasets]'

CmdStan (once):  python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
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
    """Load the Kaggle table (or LOCAL_CSV if present)."""
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
    """Outcome dec; three numeric predictors. Drop rows with missing values."""
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
    """Stratified train/test split; standardize predictors using training mean and std only."""
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
    """Compile Stan model and run MCMC."""
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
    """Print posterior summary for alpha and beta (plus global R-hat / ESS)."""
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
