from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..metrics.calibration import ReliabilityBin

COLOR_J = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"
INK_SECONDARY = "#52514e"


def plot_reliability_diagram(bins: list[ReliabilityBin], *, ece: float, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", label="Perfect calibration")

    xs = [b.mean_predicted for b in bins if b.count > 0]
    ys = [b.mean_actual for b in bins if b.count > 0]
    sizes = [20 + 300 * (b.count / max(bb.count for bb in bins)) for b in bins if b.count > 0]
    ax.scatter(xs, ys, s=sizes, color=COLOR_J, alpha=0.85, edgecolors="white", linewidths=1, zorder=3, label="Jev (per bin)")
    ax.plot(xs, ys, color=COLOR_J, linewidth=1.5, alpha=0.5, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Jev predicted probability (noul, bin mean)", color=INK_SECONDARY)
    ax.set_ylabel("Actual hallucination rate", color=INK_SECONDARY)
    ax.set_title(f"Jev calibration (ECE={ece:.3f})", color="#0b0b0b", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
