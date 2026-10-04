from __future__ import annotations

import random
from collections.abc import Callable


def bootstrap_ci(
    n: int,
    statistic_fn: Callable[[list[int]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """인덱스 리샘플링 기반 부트스트랩. statistic_fn(indices)가 통계치를 계산한다.

    반환: (원본 표본 통계치, lower, upper) — percentile method, 95% CI 기본.
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    point = statistic_fn(list(range(n)))

    samples = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(statistic_fn(idx))

    samples.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return point, samples[lo_idx], samples[hi_idx]
