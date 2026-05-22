"""Plotting helpers — QD archive heatmap, tournament matrix."""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def archive_heatmap(
    grid: np.ndarray,
    out_path: str,
    title: str = "MAP-Elites archive",
    xlabel: str = "BC dim 0",
    ylabel: str = "BC dim 1",
    bc_bounds: Optional[List] = None,
) -> str:
    """Heatmap of fitness per cell; NaN cells stay blank.

    ``bc_bounds`` (optional) is a 2-list of ``(min, max)`` ranges used to
    label the axes with real BC values rather than grid indices.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = None
    if bc_bounds is not None and len(bc_bounds) == 2:
        (x_lo, x_hi), (y_lo, y_hi) = bc_bounds
        extent = (x_lo, x_hi, y_lo, y_hi)
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("fitness")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def tournament_heatmap(
    matrix: np.ndarray,
    row_labels: Iterable[str],
    col_labels: Iterable[str],
    out_path: str,
    title: str = "Cross-play return",
    cell_metric_name: str = "predator return",
) -> str:
    """Matrix of cross-play returns (rows = predator ckpts, cols = prey ckpts)."""
    rows = list(row_labels)
    cols = list(col_labels)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(cols), 1.2 + 0.7 * len(rows)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_xlabel("prey ckpt")
    ax.set_ylabel("predator ckpt")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cell_metric_name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
