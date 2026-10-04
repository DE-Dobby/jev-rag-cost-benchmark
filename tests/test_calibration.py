from jev_rag_bench.metrics.calibration import (
    expected_calibration_error,
    reliability_bins,
)


def test_perfectly_calibrated_zero_ece():
    # 예측 확률의 평균과 실제 환각 비율이 정확히 같으면 ECE는 0
    probs = [1.0, 1.0, 1.0, 1.0]
    labels = [True, True, True, True]
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 1e-9


def test_overconfident_wrong_has_high_ece():
    probs = [0.99, 0.99, 0.99, 0.99]
    labels = [False, False, False, False]
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.9


def test_reliability_bins_counts_sum_to_total():
    probs = [0.05, 0.15, 0.55, 0.95, 0.99]
    labels = [False, False, True, True, True]
    bins = reliability_bins(probs, labels, n_bins=10)
    assert sum(b.count for b in bins) == len(probs)
