from jev_rag_bench.metrics.classification import classification_metrics


def test_perfect_predictions():
    m = classification_metrics([True, False, True, False], [True, False, True, False])
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_all_wrong():
    m = classification_metrics([True, True], [False, False])
    assert m.accuracy == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_mixed():
    # gt: T T F F, pred: T F F T -> tp=1, fp=1, fn=1, tn=1
    preds = [True, False, False, True]
    labels = [True, True, False, False]
    m = classification_metrics(preds, labels)
    assert m.accuracy == 0.5
    assert m.precision == 0.5  # tp=1, fp=1
    assert m.recall == 0.5  # tp=1, fn=1
    assert abs(m.f1 - 0.5) < 1e-9


def test_empty():
    m = classification_metrics([], [])
    assert m.n == 0
    assert m.accuracy == 0.0
