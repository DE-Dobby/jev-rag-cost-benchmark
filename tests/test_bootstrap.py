from jev_rag_bench.metrics.bootstrap import bootstrap_ci


def test_constant_values_zero_width_ci():
    values = [1.0] * 50

    def stat(idx):
        return sum(values[i] for i in idx) / len(idx)

    point, lo, hi = bootstrap_ci(len(values), stat, n_resamples=200, seed=1)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_ci_contains_point_for_varying_values():
    values = [0.0, 1.0] * 25

    def stat(idx):
        return sum(values[i] for i in idx) / len(idx)

    point, lo, hi = bootstrap_ci(len(values), stat, n_resamples=500, seed=1)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_empty_returns_zeros():
    assert bootstrap_ci(0, lambda idx: 1.0) == (0.0, 0.0, 0.0)
