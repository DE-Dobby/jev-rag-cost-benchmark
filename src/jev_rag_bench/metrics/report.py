from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from ..cascade.simulate import coverage, simulate_batch
from .bootstrap import bootstrap_ci
from .classification import classification_metrics


@dataclass
class ConfigResult:
    name: str
    accuracy: float
    accuracy_ci: tuple[float, float]
    f1: float
    f1_ci: tuple[float, float]
    precision: float
    recall: float
    cost_per_sample: float
    latency_mean_ms: float
    latency_p95_ms: float
    n: int
    coverage: float | None = None  # B(캐스케이드)에서만 의미 있음


def _p95(values: list[float]) -> float:
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


def evaluate_simple_config(
    rows: list[dict[str, Any]],
    *,
    name: str,
    pred_key: str,
    cost_key: str,
    latency_key: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> ConfigResult:
    """A/J/C처럼 단일 판정기 결과를 평가 (부트스트랩 95% CI 포함)."""
    preds = [r[pred_key] for r in rows]
    labels = [r["hallucinated_gt"] for r in rows]
    m = classification_metrics(preds, labels)

    def acc_stat(idx: list[int]) -> float:
        return classification_metrics([preds[i] for i in idx], [labels[i] for i in idx]).accuracy

    def f1_stat(idx: list[int]) -> float:
        return classification_metrics([preds[i] for i in idx], [labels[i] for i in idx]).f1

    _, acc_lo, acc_hi = bootstrap_ci(len(rows), acc_stat, n_resamples=n_bootstrap, seed=seed)
    _, f1_lo, f1_hi = bootstrap_ci(len(rows), f1_stat, n_resamples=n_bootstrap, seed=seed)

    costs = [r[cost_key] for r in rows]
    latencies = [r[latency_key] for r in rows]
    return ConfigResult(
        name=name,
        accuracy=m.accuracy,
        accuracy_ci=(acc_lo, acc_hi),
        f1=m.f1,
        f1_ci=(f1_lo, f1_hi),
        precision=m.precision,
        recall=m.recall,
        cost_per_sample=statistics.mean(costs),
        latency_mean_ms=statistics.mean(latencies),
        latency_p95_ms=_p95(latencies),
        n=m.n,
    )


def evaluate_cascade_at_threshold(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    fallback: str = "pro",
    n_bootstrap: int = 2000,
    seed: int = 42,
    name: str = "B_cascade",
) -> ConfigResult:
    outcomes = simulate_batch(rows, threshold=threshold, fallback=fallback)
    preds = [o.prediction for o in outcomes]
    labels = [r["hallucinated_gt"] for r in rows]
    m = classification_metrics(preds, labels)

    def acc_stat(idx: list[int]) -> float:
        return classification_metrics([preds[i] for i in idx], [labels[i] for i in idx]).accuracy

    def f1_stat(idx: list[int]) -> float:
        return classification_metrics([preds[i] for i in idx], [labels[i] for i in idx]).f1

    _, acc_lo, acc_hi = bootstrap_ci(len(rows), acc_stat, n_resamples=n_bootstrap, seed=seed)
    _, f1_lo, f1_hi = bootstrap_ci(len(rows), f1_stat, n_resamples=n_bootstrap, seed=seed)

    costs = [o.cost_usd for o in outcomes]
    latencies = [o.latency_ms for o in outcomes]
    return ConfigResult(
        name=name,
        accuracy=m.accuracy,
        accuracy_ci=(acc_lo, acc_hi),
        f1=m.f1,
        f1_ci=(f1_lo, f1_hi),
        precision=m.precision,
        recall=m.recall,
        cost_per_sample=statistics.mean(costs),
        latency_mean_ms=statistics.mean(latencies),
        latency_p95_ms=_p95(latencies),
        n=m.n,
        coverage=coverage(outcomes),
    )


def sweep_thresholds_on_dev(
    dev_rows: list[dict[str, Any]], thresholds: list[float], *, fallback: str = "pro"
) -> list[dict[str, Any]]:
    results = []
    for t in thresholds:
        outcomes = simulate_batch(dev_rows, threshold=t, fallback=fallback)
        preds = [o.prediction for o in outcomes]
        labels = [r["hallucinated_gt"] for r in dev_rows]
        m = classification_metrics(preds, labels)
        results.append(
            {
                "threshold": t,
                "accuracy": m.accuracy,
                "f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "cost_per_sample": statistics.mean(o.cost_usd for o in outcomes),
                "coverage": coverage(outcomes),
            }
        )
    return results


def pick_best_threshold(sweep_results: list[dict[str, Any]]) -> float:
    """F1이 가장 높은 임계값, 동률이면 비용이 더 낮은 쪽. dev에서만 수행 (test 과적합 방지)."""
    best = max(sweep_results, key=lambda r: (r["f1"], -r["cost_per_sample"]))
    return best["threshold"]


def per_task_breakdown(rows: list[dict[str, Any]], *, pred_key: str) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r)
    out = {}
    for task, trows in sorted(by_task.items()):
        preds = [r[pred_key] for r in trows]
        labels = [r["hallucinated_gt"] for r in trows]
        m = classification_metrics(preds, labels)
        out[task] = {"n": m.n, "accuracy": m.accuracy, "f1": m.f1}
    return out


def per_task_cascade_breakdown(
    rows: list[dict[str, Any]], *, threshold: float, fallback: str = "pro"
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r)
    out = {}
    for task, trows in sorted(by_task.items()):
        outcomes = simulate_batch(trows, threshold=threshold, fallback=fallback)
        preds = [o.prediction for o in outcomes]
        labels = [r["hallucinated_gt"] for r in trows]
        m = classification_metrics(preds, labels)
        out[task] = {"n": m.n, "accuracy": m.accuracy, "f1": m.f1, "coverage": coverage(outcomes)}
    return out


def cost_by_length_bins(
    rows: list[dict[str, Any]],
    *,
    token_lengths: dict[str, int],
    threshold: float,
    n_bins: int = 4,
) -> list[dict[str, Any]]:
    """근거 문서 토큰 길이 구간별로 A(Pro단독) vs B(캐스케이드) 평균 비용 비교."""
    enriched = [(r, token_lengths[r["response_id"]]) for r in rows if r["response_id"] in token_lengths]
    enriched.sort(key=lambda t: t[1])
    n = len(enriched)
    if n == 0:
        return []

    outcomes = {
        r["response_id"]: o for r, o in zip(rows, simulate_batch(rows, threshold=threshold))
    }

    bins = []
    for i in range(n_bins):
        lo = i * n // n_bins
        hi = (i + 1) * n // n_bins if i < n_bins - 1 else n
        chunk = enriched[lo:hi]
        if not chunk:
            continue
        a_cost = statistics.mean(r["pro_cost_usd"] for r, _ in chunk)
        b_cost = statistics.mean(outcomes[r["response_id"]].cost_usd for r, _ in chunk)
        bins.append(
            {
                "token_range": (chunk[0][1], chunk[-1][1]),
                "n": len(chunk),
                "a_cost_per_sample": a_cost,
                "b_cost_per_sample": b_cost,
            }
        )
    return bins


def breakeven_table(sweep_results: list[dict[str, Any]], *, jev_cost_per_sample: float, pro_cost_per_sample: float) -> list[dict[str, Any]]:
    """c_jev < coverage * c_llm 조건이 성립하는지 임계값별로 표시."""
    out = []
    for r in sweep_results:
        holds = jev_cost_per_sample < r["coverage"] * pro_cost_per_sample
        out.append({"threshold": r["threshold"], "coverage": r["coverage"], "breakeven_holds": holds})
    return out
