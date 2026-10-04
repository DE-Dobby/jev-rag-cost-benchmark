from jev_rag_bench.metrics.report import breakeven_table, pick_best_threshold


def test_pick_best_threshold_prefers_highest_f1():
    sweep = [
        {"threshold": 0.5, "f1": 0.70, "cost_per_sample": 0.001},
        {"threshold": 0.9, "f1": 0.85, "cost_per_sample": 0.002},
        {"threshold": 0.99, "f1": 0.80, "cost_per_sample": 0.003},
    ]
    assert pick_best_threshold(sweep) == 0.9


def test_pick_best_threshold_tie_breaks_on_lower_cost():
    sweep = [
        {"threshold": 0.7, "f1": 0.80, "cost_per_sample": 0.005},
        {"threshold": 0.9, "f1": 0.80, "cost_per_sample": 0.002},
    ]
    assert pick_best_threshold(sweep) == 0.9


def test_breakeven_holds_when_jev_much_cheaper():
    sweep = [{"threshold": 0.9, "coverage": 0.5, "f1": 0.8, "cost_per_sample": 0.001}]
    out = breakeven_table(sweep, jev_cost_per_sample=0.0001, pro_cost_per_sample=0.002)
    assert out[0]["breakeven_holds"] is True


def test_breakeven_fails_when_coverage_too_low():
    sweep = [{"threshold": 0.99, "coverage": 0.01, "f1": 0.8, "cost_per_sample": 0.001}]
    out = breakeven_table(sweep, jev_cost_per_sample=0.0001, pro_cost_per_sample=0.002)
    # coverage*pro = 0.01*0.002 = 0.00002 < jev_cost 0.0001 -> breakeven 깨짐
    assert out[0]["breakeven_holds"] is False
