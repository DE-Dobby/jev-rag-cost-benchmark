from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CascadeOutcome:
    prediction: bool
    cost_usd: float
    latency_ms: float
    fell_back: bool


def simulate_row(row: dict[str, Any], *, threshold: float, fallback: str = "pro") -> CascadeOutcome:
    """row는 to_combined_rows()가 만든 형태 (jev_noul, jev_hallucinated, jev_cost_usd, jev_latency_ms,
    {fallback}_hallucinated, {fallback}_cost_usd, {fallback}_latency_ms 등을 가진 dict).
    Jev 신뢰도 = max(p, 1-p). fallback은 "pro" 또는 "flash" — 캐스케이드의 폴백 모델을 고른다."""
    noul = row["jev_noul"]
    confidence = max(noul, 1 - noul)
    cost = row["jev_cost_usd"]
    latency = row["jev_latency_ms"]

    if confidence >= threshold:
        return CascadeOutcome(prediction=row["jev_hallucinated"], cost_usd=cost, latency_ms=latency, fell_back=False)

    cost += row[f"{fallback}_cost_usd"]
    latency += row[f"{fallback}_latency_ms"]
    return CascadeOutcome(
        prediction=row[f"{fallback}_hallucinated"], cost_usd=cost, latency_ms=latency, fell_back=True
    )


def simulate_batch(rows: list[dict[str, Any]], *, threshold: float, fallback: str = "pro") -> list[CascadeOutcome]:
    return [simulate_row(r, threshold=threshold, fallback=fallback) for r in rows]


def coverage(outcomes: list[CascadeOutcome]) -> float:
    """Jev가 (폴백 없이) 직접 처리한 비율."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if not o.fell_back) / len(outcomes)
