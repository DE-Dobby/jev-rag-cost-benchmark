from __future__ import annotations

import numpy as np

from .loader import grounding_text
from .schema import MergedSample
from .tokens import approx_token_count


def _quantiles(values: list[int]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def build_stats(samples: list[MergedSample]) -> dict:
    n = len(samples)
    by_task: dict[str, list[MergedSample]] = {}
    by_split: dict[str, list[MergedSample]] = {}
    for s in samples:
        by_task.setdefault(s.task_type, []).append(s)
        by_split.setdefault(s.split, []).append(s)

    token_counts = [approx_token_count(grounding_text(s)) for s in samples]

    task_stats = {}
    for task, items in by_task.items():
        halluc = sum(1 for s in items if s.hallucinated)
        task_tokens = [approx_token_count(grounding_text(s)) for s in items]
        task_stats[task] = {
            "n": len(items),
            "hallucination_rate": halluc / len(items),
            "grounding_tokens": _quantiles(task_tokens),
        }

    split_stats = {}
    for split, items in by_split.items():
        halluc = sum(1 for s in items if s.hallucinated)
        split_stats[split] = {
            "n": len(items),
            "hallucination_rate": halluc / len(items),
        }

    quality_counts: dict[str, int] = {}
    for s in samples:
        quality_counts[s.quality] = quality_counts.get(s.quality, 0) + 1

    model_counts: dict[str, int] = {}
    for s in samples:
        model_counts[s.model] = model_counts.get(s.model, 0) + 1

    return {
        "n_total": n,
        "hallucination_rate_overall": sum(1 for s in samples if s.hallucinated) / n if n else 0.0,
        "by_task": task_stats,
        "by_split": split_stats,
        "quality_counts": quality_counts,
        "model_counts": model_counts,
        "grounding_tokens_overall": _quantiles(token_counts),
    }


def format_stats(stats: dict) -> str:
    lines: list[str] = []
    lines.append(f"전체 샘플 수: {stats['n_total']}")
    lines.append(f"전체 환각 비율: {stats['hallucination_rate_overall']:.3f}")
    lines.append("")
    lines.append("[태스크별]")
    for task, s in stats["by_task"].items():
        lines.append(f"  {task}: n={s['n']}, 환각비율={s['hallucination_rate']:.3f}, "
                      f"근거문서토큰(mean/p50/p95)={s['grounding_tokens']['mean']:.0f}/"
                      f"{s['grounding_tokens']['p50']:.0f}/{s['grounding_tokens']['p95']:.0f}")
    lines.append("")
    lines.append("[split별]")
    for split, s in stats["by_split"].items():
        lines.append(f"  {split}: n={s['n']}, 환각비율={s['hallucination_rate']:.3f}")
    lines.append("")
    lines.append(f"[quality 분포] {stats['quality_counts']}")
    lines.append(f"[model 분포] {stats['model_counts']}")
    lines.append("")
    gt = stats["grounding_tokens_overall"]
    lines.append(
        "[근거 문서 토큰 길이(전체, cl100k_base 근사치)] "
        f"mean={gt['mean']:.0f} p50={gt['p50']:.0f} p90={gt['p90']:.0f} "
        f"p95={gt['p95']:.0f} p99={gt['p99']:.0f} max={gt['max']:.0f}"
    )
    return "\n".join(lines)
