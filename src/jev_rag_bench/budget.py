from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    def __init__(self, spent_usd: float, max_usd: float):
        self.spent_usd = spent_usd
        self.max_usd = max_usd
        super().__init__(f"예산 초과: ${spent_usd:.4f} > ${max_usd:.4f}")


class BudgetGuard:
    """누적 비용을 추적하고 상한(max_usd)을 넘으면 즉시 중단시킨다. 스레드 세이프."""

    def __init__(self, max_usd: float):
        self._max_usd = max_usd
        self._spent_usd = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    def add(self, usd: float) -> None:
        with self._lock:
            self._spent_usd += usd
            if self._spent_usd > self._max_usd:
                raise BudgetExceeded(self._spent_usd, self._max_usd)
