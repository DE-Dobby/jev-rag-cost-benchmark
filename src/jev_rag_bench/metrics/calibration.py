from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReliabilityBin:
    mean_predicted: float | None  # 이 구간 노울 값들의 평균
    mean_actual: float | None  # 실제 환각 비율
    count: int


def reliability_bins(probs: list[float], labels: list[bool], *, n_bins: int = 10) -> list[ReliabilityBin]:
    """노울 확률(환각일 확률)을 n_bins개 구간으로 나눠 (예측 평균, 실제 환각 비율) 계산."""
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(probs) if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
        if not idx:
            bins.append(ReliabilityBin(mean_predicted=None, mean_actual=None, count=0))
            continue
        mean_pred = sum(probs[j] for j in idx) / len(idx)
        mean_actual = sum(1 for j in idx if labels[j]) / len(idx)
        bins.append(ReliabilityBin(mean_predicted=mean_pred, mean_actual=mean_actual, count=len(idx)))
    return bins


def expected_calibration_error(probs: list[float], labels: list[bool], *, n_bins: int = 10) -> float:
    n = len(probs)
    if n == 0:
        return 0.0
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    ece = 0.0
    for b in bins:
        if b.count == 0:
            continue
        ece += (b.count / n) * abs(b.mean_predicted - b.mean_actual)
    return ece
