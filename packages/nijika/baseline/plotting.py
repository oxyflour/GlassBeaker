from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline.metrics import magnitude_db


def save_matrix_plot(
    path: Path,
    freq_grid: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
    title: str,
    port_count: int,
) -> None:
    pred_db = magnitude_db(pred.reshape(len(freq_grid), port_count, port_count, 2))
    truth_db = None
    if truth is not None:
        truth_db = magnitude_db(truth.reshape(len(freq_grid), port_count, port_count, 2))
    fig, axes = plt.subplots(port_count, port_count, figsize=(12, 10), sharex=True)
    ghz = freq_grid
    for row in range(port_count):
        for col in range(port_count):
            axis = axes[row, col]
            if truth_db is not None:
                axis.plot(ghz, truth_db[:, row, col], label="true", linewidth=1.5)
            axis.plot(ghz, pred_db[:, row, col], label="pred", linewidth=1.2, linestyle="--")
            axis.set_title(f"S{row + 1}{col + 1}")
            axis.grid(alpha=0.2)
            if row == port_count - 1:
                axis.set_xlabel("GHz")
            if col == 0:
                axis.set_ylabel("dB")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
