"""Problem 5: simulate data, fit Stan model"""

from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

SEED = 12345
TRUE_ALPHA = 10.0
TRUE_BETA = 3.0
TRUE_SIGMA = 3.0


def simulate_data(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=n)
    y = TRUE_ALPHA + TRUE_BETA * x + TRUE_SIGMA * rng.normal(size=n)
    return x, y


def fit_model(model: CmdStanModel, x: np.ndarray, y: np.ndarray, seed: int):
    return model.sample(
        data={"N": len(x), "x": x.tolist(), "y": y.tolist()},
        seed=seed,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
    )


def summarize_fit(fit, n: int) -> Dict[str, float]:
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

    a = stats(posterior["alpha"])
    b = stats(posterior["beta"])
    s = stats(posterior["sigma"])

    diag_rows = [p for p in ["alpha", "beta", "sigma2"] if p in summary.index]
    diag_cols = [c for c in ["R_hat", "ESS_bulk", "ESS_tail"] if c in summary.columns]
    diag = summary.loc[diag_rows, diag_cols]
    worst_rhat = float(diag["R_hat"].max()) if "R_hat" in diag.columns else float("nan")
    min_ess = float(diag["ESS_bulk"].min()) if "ESS_bulk" in diag.columns else float("nan")

    return {
        "N": n,
        "alpha_mean": a["mean"],
        "beta_mean": b["mean"],
        "sigma_mean": s["mean"],
        "alpha_sd": a["sd"],
        "beta_sd": b["sd"],
        "sigma_sd": s["sd"],
        "alpha_2p5": a["q2p5"],
        "alpha_97p5": a["q97p5"],
        "beta_2p5": b["q2p5"],
        "beta_97p5": b["q97p5"],
        "sigma_2p5": s["q2p5"],
        "sigma_97p5": s["q97p5"],
        "max_rhat": worst_rhat,
        "min_ess_bulk": min_ess,
    }


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent
    out_dir = hw3_dir / "problem5_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    model = CmdStanModel(stan_file=str(hw3_dir / "problem5_linear_regression.stan"))

    rows = []
    for n in (100, 1000):
        x, y = simulate_data(n, rng)
        fit = fit_model(model, x, y, seed=SEED + n)
        rows.append(summarize_fit(fit, n))

    df = pd.DataFrame(rows).sort_values("N")
    df.to_csv(out_dir / "problem5_summary.csv", index=False)
    (out_dir / "problem5_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
