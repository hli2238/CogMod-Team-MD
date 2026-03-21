# Problem 5: Simple Bayesian Regression with Stan
# Michael Lam and Dillon Li
# March 2026
# Usage: python3 problem5


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cmdstanpy import CmdStanModel

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd


"""
I set the parameters of the linear model to be:
alpha = 10.0
beta = 3.0
sigma = 3.0
"""
SEED = 12345
TRUE_ALPHA = 10.0  
TRUE_BETA = 3.0  
TRUE_SIGMA = 3.0   


def simulate_data(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data to test our model from the linear model y = alpha + beta*x + eps.

    Args:
        n: Number of observations.
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of (x, y) arrays; x ~ N(0,1), y ~ N(alpha + beta*x, sigma^2).
    """
    x = rng.normal(size=n)
    y = TRUE_ALPHA + TRUE_BETA * x + TRUE_SIGMA * rng.normal(size=n)
    return x, y


def fit_model(model: CmdStanModel, x: np.ndarray, y: np.ndarray, seed: int):
    """Run MCMC sampling for the Bayesian linear regression model.

    Args:
        model: Compiled CmdStanModel (problem5_linear_regression.stan).
        x: Covariate values.
        y: Outcome values.
        seed: Random seed for reproducibility.

    Returns:
        CmdStanMCMC fit object with posterior draws.
    """
    fit = model.sample(
        data={"N": len(x), "x": x.tolist(), "y": y.tolist()},
        seed=seed,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        refresh=250,
    )
    return fit


def summarize_fit(fit, n: int) -> Dict[str, float]:
    """Extract posterior summaries from the Stan fit.

    Args:
        fit: CmdStanMCMC fit object.
        n: Sample size.

    Returns:
        Dict with posterior stats, diagnostics, and abs errors vs. TRUE_*.
    """
    summary = fit.summary()
    posterior = fit.draws_pd(vars=["alpha", "beta", "sigma"])

    def stats(series: pd.Series) -> Dict[str, float]:
        arr = series.to_numpy()
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)),
            "q2p5": float(np.quantile(arr, 0.025)),
            "q97p5": float(np.quantile(arr, 0.975)),
        }

    alpha_stats = stats(posterior["alpha"])
    beta_stats = stats(posterior["beta"])
    sigma_stats = stats(posterior["sigma"])

    alpha_mean = alpha_stats["mean"]
    beta_mean = beta_stats["mean"]
    sigma_mean = sigma_stats["mean"]
    alpha_err = abs(alpha_mean - TRUE_ALPHA)
    beta_err = abs(beta_mean - TRUE_BETA)
    sigma_err = abs(sigma_mean - TRUE_SIGMA)

    diag_rows = [p for p in ["alpha", "beta", "sigma2"] if p in summary.index]
    diag_cols = [c for c in ["R_hat", "ESS_bulk", "ESS_tail"] if c in summary.columns]
    diag = summary.loc[diag_rows, diag_cols]

    print("Summary of posterior: ")
    print(
        pd.DataFrame(
            {
                "Mean": [alpha_mean, beta_mean, sigma_mean],
                "SD": [alpha_stats["sd"], beta_stats["sd"], sigma_stats["sd"]],
                "2.5%": [alpha_stats["q2p5"], beta_stats["q2p5"], sigma_stats["q2p5"]],
                "97.5%": [alpha_stats["q97p5"], beta_stats["q97p5"], sigma_stats["q97p5"]],
            },
            index=["alpha", "beta", "sigma"],
        ).to_string(float_format=lambda v: f"{v:0.4f}")
    )
    if not diag.empty:
        print("\nDiagnostics:")
        print(diag.to_string(float_format=lambda v: f"{v:0.4f}"))
    print(
        f"\nAbsolute posterior-mean error: "
        f"alpha={alpha_err:0.4f}, beta={beta_err:0.4f}, sigma={sigma_err:0.4f}"
    )

    worst_rhat = float(diag["R_hat"].max()) if "R_hat" in diag.columns else float("nan")
    min_ess = float(diag["ESS_bulk"].min()) if "ESS_bulk" in diag.columns else float("nan")

    return {
        "N": n,
        "alpha_mean": alpha_mean,
        "beta_mean": beta_mean,
        "sigma_mean": sigma_mean,
        "alpha_sd": alpha_stats["sd"],
        "beta_sd": beta_stats["sd"],
        "sigma_sd": sigma_stats["sd"],
        "alpha_2p5": alpha_stats["q2p5"],
        "alpha_97p5": alpha_stats["q97p5"],
        "beta_2p5": beta_stats["q2p5"],
        "beta_97p5": beta_stats["q97p5"],
        "sigma_2p5": sigma_stats["q2p5"],
        "sigma_97p5": sigma_stats["q97p5"],
        "max_rhat": worst_rhat,
        "min_ess_bulk": min_ess,
        "alpha_abs_error": alpha_err,
        "beta_abs_error": beta_err,
        "sigma_abs_error": sigma_err,
    }


def make_plots(fit, x: np.ndarray, y: np.ndarray, n: int, out_dir: Path) -> None:
    """Organizes the data and plots the posterior marginals, regression fit, and trace plots. 

    Also saves the plots to the output directory.

    Args:
        fit: CmdStanMCMC fit object.
        x: Covariate values.
        y: Outcome values.
        n: Sample size.
        out_dir: Directory to save figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    posterior = fit.draws_pd(vars=["alpha", "beta", "sigma"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name, truth in zip(
        axes,
        ["alpha", "beta", "sigma"],
        [TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA],
    ):
        ax.hist(posterior[name], bins=40, alpha=0.8)
        ax.axvline(truth, linestyle="--")
        ax.set_title(f"{name} posterior")
    fig.suptitle(f"Posterior marginals (N={n})")
    fig.tight_layout()
    fig.savefig(out_dir / f"problem5_posteriors_N{n}.png", dpi=150)
    plt.close(fig)

    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    alpha_s = posterior["alpha"].to_numpy()
    beta_s = posterior["beta"].to_numpy()
    idx = np.linspace(0, len(alpha_s) - 1, min(200, len(alpha_s)), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=12, alpha=0.5, label="data")
    for i in idx:
        mu = alpha_s[i] + beta_s[i] * grid
        ax.plot(grid, mu, color="tab:blue", alpha=0.04)
    mu_mean = float(np.mean(alpha_s)) + float(np.mean(beta_s)) * grid
    ax.plot(grid, mu_mean, color="black", linewidth=2, label="posterior mean line")
    ax.plot(grid, TRUE_ALPHA + TRUE_BETA * grid, linestyle="--", color="tab:red", label="true line")
    ax.set_title(f"Regression fit (N={n})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"problem5_fit_N{n}.png", dpi=150)
    plt.close(fig)

    idata = az.from_cmdstanpy(posterior=fit, posterior_predictive="y_rep")
    az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])
    plt.tight_layout()
    plt.savefig(out_dir / f"problem5_trace_N{n}.png", dpi=150)
    plt.close()


def main() -> None:
    """Orchestrate simulation, fitting, summarization, and saving for N=100 and N=1000."""
    hw3_dir = Path(__file__).resolve().parent
    stan_path = hw3_dir / "problem5_linear_regression.stan"
    out_dir = hw3_dir / "problem5_outputs"

    rng = np.random.default_rng(SEED)
    model = CmdStanModel(stan_file=str(stan_path))

    rows = []
    for n in [100, 1000]:
        x, y = simulate_data(n, rng)
        fit = fit_model(model, x, y, seed=SEED + n)
        rows.append(summarize_fit(fit, n))
        make_plots(fit, x, y, n, out_dir)

    df = pd.DataFrame(rows).sort_values("N")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "problem5_summary.csv", index=False)
    (out_dir / "problem5_summary.json").write_text(json.dumps(rows, indent=2))

    print("\nSaved outputs:")
    print(f"- {out_dir / 'problem5_summary.csv'}")
    print(f"- {out_dir / 'problem5_summary.json'}")
    print(f"- {out_dir / 'problem5_posteriors_N100.png'}")
    print(f"- {out_dir / 'problem5_posteriors_N1000.png'}")
    print(f"- {out_dir / 'problem5_fit_N100.png'}")
    print(f"- {out_dir / 'problem5_fit_N1000.png'}")
    print(f"- {out_dir / 'problem5_trace_N100.png'}")
    print(f"- {out_dir / 'problem5_trace_N1000.png'}")

if __name__ == "__main__":
    main()

