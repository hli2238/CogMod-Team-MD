from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


def prepare_data(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Stan data dict and condition metadata from a trial-level dataframe.
    Args:
        df: Must have columns ``id``, ``rt``, ``choice``, ``condition``.

    Returns:
        A pair ``(stan_data, meta)`` where ``stan_data`` keys are ``N``, ``J``,
        ``id``, ``y``, ``condition``, ``choice`` (lists for Stan), and ``meta``
        contains ``condition_map_stan_to_original`` for decoding results.
    """
    id_values = sorted(df["id"].unique().tolist())
    id_map = {v: i + 1 for i, v in enumerate(id_values)}
    df = df.copy()
    df["id_stan"] = df["id"].map(id_map).astype(int)

    cond_values = sorted(df["condition"].unique().tolist())
    cond_map = {cond_values[0]: 1, cond_values[1]: 2}
    inv_cond_map = {1: cond_values[0], 2: cond_values[1]}
    df["condition_stan"] = df["condition"].map(cond_map).astype(int)

    stan_data: dict[str, Any] = {
        "N": len(df),
        "J": len(id_values),
        "id": df["id_stan"].astype(int).tolist(),
        "y": df["rt"].astype(float).tolist(),
        "condition": df["condition_stan"].astype(int).tolist(),
        "choice": df["choice"].astype(int).tolist(),
    }
    meta = {"condition_map_stan_to_original": inv_cond_map}
    return stan_data, meta


def summarize_result(fit: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Summarize posterior for mean drift by condition and identify harder field.

    Args:
        fit: A ``CmdStanMCMC`` object from ``model.sample(...)``.
        meta: Output of ``prepare_data``; needs ``condition_map_stan_to_original``.

    Returns:
        Dict with posterior means, 95% interval for drift difference, ``R_hat`` /
        ESS summaries, and ``hard_condition_original_code`` (CSV condition value).
    """
    draws = fit.draws_pd(vars=["mean_v_cond1", "mean_v_cond2", "diff_v_2_minus_1"])
    summary = fit.summary()

    m1 = draws["mean_v_cond1"]
    m2 = draws["mean_v_cond2"]
    d = draws["diff_v_2_minus_1"]

    hard_stan = 1 if m1.mean() < m2.mean() else 2
    inv = meta["condition_map_stan_to_original"]

    return {
        "mean_v_cond1_mean": float(m1.mean()),
        "mean_v_cond2_mean": float(m2.mean()),
        "diff_v_2_minus_1_mean": float(d.mean()),
        "diff_v_2_minus_1_2p5": float(np.quantile(d.to_numpy(), 0.025)),
        "diff_v_2_minus_1_97p5": float(np.quantile(d.to_numpy(), 0.975)),
        "max_rhat": float(summary["R_hat"].max()),
        "min_ess_bulk": float(summary["ESS_bulk"].min()),
        "hard_condition_original_code": inv[hard_stan],
    }


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent
    out_dir = hw3_dir / "problem6_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = next(hw3_dir.glob("*response*times*.csv"))
    first = csv_path.read_text(encoding="utf-8").splitlines()[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    df = pd.read_csv(csv_path, sep=sep)

    stan_data, meta = prepare_data(df)
    model = CmdStanModel(stan_file=str(hw3_dir / "diffusion_multiple.stan"))
    fit = model.sample(
        data=stan_data,
        seed=12345,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        refresh=250,
    )

    result = summarize_result(fit, meta)
    result["data_file"] = csv_path.name
    result["N"] = stan_data["N"]
    result["J"] = stan_data["J"]

    (out_dir / "problem6_summary.json").write_text(json.dumps(result, indent=2))
    fit.summary().to_csv(out_dir / "problem6_cmdstan_summary.csv")

    print("\n=== Problem 6 result summary ===")
    print(f"Data file: {result['data_file']}")
    print(f"N={result['N']}, J={result['J']}")
    print(
        f"Posterior mean drift: cond1={result['mean_v_cond1_mean']:.4f}, "
        f"cond2={result['mean_v_cond2_mean']:.4f}"
    )
    print(
        f"Difference (cond2-cond1): mean={result['diff_v_2_minus_1_mean']:.4f}, "
        f"95% CrI=[{result['diff_v_2_minus_1_2p5']:.4f}, {result['diff_v_2_minus_1_97p5']:.4f}]"
    )
    print(f"max R_hat={result['max_rhat']:.4f}, min ESS_bulk={result['min_ess_bulk']:.2f}")
    print(f"High-interference condition (original code): {result['hard_condition_original_code']}")


if __name__ == "__main__":
    main()
