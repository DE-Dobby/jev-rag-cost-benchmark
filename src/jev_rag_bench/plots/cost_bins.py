from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .pareto import COLOR_A, COLOR_B, GRID, INK_SECONDARY, MUTED


def plot_cost_by_length_bins(bins: list[dict[str, Any]], *, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    labels = [f"{b['token_range'][0]}-{b['token_range'][1]}" for b in bins]
    a_vals = [b["a_cost_per_sample"] for b in bins]
    b_vals = [b["b_cost_per_sample"] for b in bins]

    x = range(len(bins))
    width = 0.36
    ax.bar([i - width / 2 for i in x], a_vals, width=width, color=COLOR_A, label="A: Gemini Pro alone")
    ax.bar([i + width / 2 for i in x], b_vals, width=width, color=COLOR_B, label="B: cascade (t=0.90)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Context length (tokens, approx.)", color=INK_SECONDARY)
    ax.set_ylabel("Cost per sample (USD)", color=INK_SECONDARY)
    ax.set_title("Cost by context length — A vs B", color="#0b0b0b", fontsize=11)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
