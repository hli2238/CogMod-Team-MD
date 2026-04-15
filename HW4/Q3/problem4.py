"""HW4 Problem 4: Brier / accuracy from ``problem3_fit_meta.npz`` (no new sampling)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parent / "problem3_fit_meta.npz"


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def accuracy_at_half(p: np.ndarray, y: np.ndarray) -> float:
    """Accuracy when predicting 1 iff ``p >= 0.5``."""
    y = np.asarray(y, dtype=np.int64)
    return float(np.mean((p >= 0.5).astype(np.int64) == y))


def main() -> None:
    """Load npz, print one line of metrics (two with baselines if npz has train fields)."""
    if not NPZ.is_file():
        print(f"missing {NPZ} (run problem3_speed_dating.py)", file=sys.stderr)
        sys.exit(1)

    z = np.load(NPZ, allow_pickle=True)
    y = np.asarray(z["y_test"], dtype=np.int64).ravel()
    ph = z["p_hat_samples"]  # (S, M): draws x test rows
    s, m = ph.shape
    if y.size != m:
        sys.exit(f"y_test len {y.size} != p_hat cols {m}")

    p_bar = ph.mean(axis=0)
    b0 = brier_score(p_bar, y)
    a0 = accuracy_at_half(p_bar, y)

    # Per-draw: Brier using one posterior draw's vector p_hat[s], not mean(p_hat).
    b_s = np.mean((ph - y) ** 2, axis=1)
    q05, q50, q95 = np.quantile(b_s, [0.05, 0.5, 0.95])

    ok = "y_train_positive_rate" in z and "train_majority_label" in z
    if not ok:
        print("p4: rerun problem3_speed_dating.py to embed train baselines in npz", file=sys.stderr)
        tail = ""
    else:
        pr = float(z["y_train_positive_rate"])
        maj = int(z["train_majority_label"].ravel()[0])
        b_const = brier_score(np.full(m, pr), y)
        a_maj = float(np.mean(np.full(m, maj) == y))
        b_deg = brier_score(np.full(m, float(maj)), y)
        tail = (
            f" || base p_tr={pr:.4f} Brier_const={b_const:.4f} acc_maj({maj})={a_maj:.4f} "
            f"Brier_01={b_deg:.4f}"
        )

    print(
        f"p4 M={m} S={s} Brier={b0:.4f} acc={a0:.4f} "
        f"Brier_draw_mean={b_s.mean():.4f} q05,q50,q95={q05:.4f},{q50:.4f},{q95:.4f}{tail}"
    )


if __name__ == "__main__":
    main()
