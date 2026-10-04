from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# dataviz 스킬의 검증된 카테고리 팔레트 (고정 슬롯 순서로 사용 — 순서를 바꾸지 않음)
COLOR_A = "#2a78d6"  # blue   — A: Gemini Pro 단독 (기준)
COLOR_J = "#1baf7a"  # aqua   — J: Jev 단독
COLOR_C = "#eda100"  # yellow — C: Gemini Flash 단독
COLOR_B = "#4a3aa7"  # violet — B: 캐스케이드 (Jev+Pro)
COLOR_B2 = "#e34948"  # red    — B2: 캐스케이드 (Jev+Flash)
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
MUTED = "#898781"


def plot_cost_vs_f1(
    *,
    dev_sweep: list[dict[str, Any]],
    point_a: dict[str, Any],
    point_j: dict[str, Any],
    point_c: dict[str, Any],
    point_b_test: dict[str, Any],
    out_path: str | Path,
    point_b2_test: dict[str, Any] | None = None,
) -> None:
    """비용(x, $/sample) vs F1(y) Pareto 플롯. A/J/C는 점, B는 dev 스윕 곡선 + test 확정점."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # B: dev에서 스윕한 전체 곡선 (탐색적 — 참고용, 옅은 선 + 마커)
    xs = [r["cost_per_sample"] for r in dev_sweep]
    ys = [r["f1"] for r in dev_sweep]
    ax.plot(xs, ys, color=COLOR_B, linewidth=1.5, alpha=0.45, zorder=2, label="B: cascade (dev threshold sweep)")
    ax.scatter(xs, ys, color=COLOR_B, s=18, alpha=0.45, zorder=2)

    # B: test에 고정 적용한 확정 1점 (하이라이트)
    ax.scatter(
        [point_b_test["cost_per_sample"]],
        [point_b_test["f1"]],
        color=COLOR_B,
        s=140,
        marker="D",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label=f"B: cascade Jev+Pro (test, t={point_b_test['threshold']:.2f})",
    )

    if point_b2_test is not None:
        ax.scatter(
            [point_b2_test["cost_per_sample"]],
            [point_b2_test["f1"]],
            color=COLOR_B2,
            s=140,
            marker="*",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
            label=f"B2: cascade Jev+Flash (test, t={point_b2_test['threshold']:.2f})",
        )
        ax.annotate(
            "B2",
            (point_b2_test["cost_per_sample"], point_b2_test["f1"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=INK_SECONDARY,
            fontweight="bold",
        )

    for point, color, label in [
        (point_a, COLOR_A, "A: Gemini Pro alone"),
        (point_j, COLOR_J, "J: Jev alone"),
        (point_c, COLOR_C, "C: Gemini Flash alone"),
    ]:
        ax.scatter(
            [point["cost_per_sample"]],
            [point["f1"]],
            color=color,
            s=110,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            zorder=4,
            label=label,
        )
        ax.annotate(
            label.split(":")[0],
            (point["cost_per_sample"], point["f1"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.annotate(
        "B",
        (point_b_test["cost_per_sample"], point_b_test["f1"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
        color=INK_SECONDARY,
        fontweight="bold",
    )

    ax.set_xlabel("Cost per sample (USD)", color=INK_SECONDARY)
    ax.set_ylabel("F1 (hallucination class)", color=INK_SECONDARY)
    ax.set_title("Cost vs F1 — A/J/C are fixed points, B sweeps thresholds on dev", color="#0b0b0b", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
