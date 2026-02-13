from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_results(
    infected: Sequence[float] | npt.NDArray[np.float64],
    *,
    title: str = "Simulated Outbreak",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the time series of infected cases."""
    infected_arr = np.asarray(infected, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(infected_arr, linestyle="dashed", marker="o")
    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Number of Infected Cases", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(alpha=0.2)
    return fig
