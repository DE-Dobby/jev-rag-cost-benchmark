from jev_rag_bench.cascade.simulate import coverage, simulate_batch, simulate_row


def _row(noul, jev_pred, pro_pred):
    return {
        "jev_noul": noul,
        "jev_hallucinated": jev_pred,
        "jev_cost_usd": 0.001,
        "jev_latency_ms": 10.0,
        "pro_hallucinated": pro_pred,
        "pro_cost_usd": 0.1,
        "pro_latency_ms": 1000.0,
    }


def test_high_confidence_uses_jev():
    row = _row(0.95, True, False)
    out = simulate_row(row, threshold=0.9)
    assert out.prediction is True
    assert out.fell_back is False
    assert out.cost_usd == 0.001


def test_low_confidence_falls_back_to_pro():
    row = _row(0.55, True, False)
    out = simulate_row(row, threshold=0.9)
    assert out.prediction is False
    assert out.fell_back is True
    assert out.cost_usd == 0.001 + 0.1
    assert out.latency_ms == 10.0 + 1000.0


def test_confidence_is_symmetric_around_half():
    # noul=0.05 -> confidence = max(0.05, 0.95) = 0.95, 높은 신뢰도로 "환각 아님" 판정
    row = _row(0.05, False, True)
    out = simulate_row(row, threshold=0.9)
    assert out.fell_back is False
    assert out.prediction is False


def test_coverage():
    rows = [_row(0.95, True, False), _row(0.55, True, False), _row(0.99, False, False)]
    outcomes = simulate_batch(rows, threshold=0.9)
    assert coverage(outcomes) == 2 / 3
