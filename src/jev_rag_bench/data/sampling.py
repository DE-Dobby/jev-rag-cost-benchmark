from __future__ import annotations

import random

from .schema import MergedSample


def stratified_sample_by_task(
    samples: list[MergedSample], *, n_per_task: int, seed: int
) -> list[MergedSample]:
    """task_type별로 동일 개수를 무작위로 뽑는다 (재현 가능한 시드)."""
    rng = random.Random(seed)
    by_task: dict[str, list[MergedSample]] = {}
    for s in samples:
        by_task.setdefault(s.task_type, []).append(s)

    result: list[MergedSample] = []
    for task, items in sorted(by_task.items()):
        k = min(n_per_task, len(items))
        result.extend(rng.sample(items, k))
    return result


def simple_random_sample(samples: list[MergedSample], *, n: int, seed: int) -> list[MergedSample]:
    """단순 무작위 샘플 (태스크 비율을 원본 분포 그대로 유지 — dev/test 본 실행용)."""
    rng = random.Random(seed)
    k = min(n, len(samples))
    return rng.sample(samples, k)
