from src.eval.risk_coverage import (
    GoldenItem,
    cost_optimal_threshold,
    coverage_at_safety_target,
    risk_coverage_curve,
)


def make_items():
    # 4 confident+safe, 2 confident+unsafe, 4 low-confidence+safe
    return (
        [GoldenItem(confidence=0.9, is_safe_if_auto=True) for _ in range(4)]
        + [GoldenItem(confidence=0.85, is_safe_if_auto=False) for _ in range(2)]
        + [GoldenItem(confidence=0.1, is_safe_if_auto=True) for _ in range(4)]
    )


def test_risk_coverage_curve_monotonic_coverage():
    items = make_items()
    curve = risk_coverage_curve(items, n_thresholds=11)
    coverages = [p["coverage"] for p in curve]
    # coverage must be non-increasing as threshold rises
    assert all(coverages[i] >= coverages[i + 1] - 1e-9 for i in range(len(coverages) - 1))


def test_coverage_at_safety_target_excludes_unsafe_batch():
    items = make_items()
    curve = risk_coverage_curve(items, n_thresholds=101)
    result = coverage_at_safety_target(curve, target=1.0)
    # threshold must exceed 0.85 to exclude the two unsafe items entirely
    auto_count = round(result["coverage"] * len(items))
    assert auto_count == 4  # only the confident+safe ones


def test_cost_optimal_threshold_prefers_escalation_when_bad_auto_is_expensive():
    items = make_items()
    best_expensive = cost_optimal_threshold(items, cost_bad_auto=100.0, cost_human=1.0)
    best_cheap = cost_optimal_threshold(items, cost_bad_auto=1.0, cost_human=100.0)
    # when bad-auto is very expensive, threshold should be high (favour escalation)
    # when human touch is very expensive, threshold should be low (favour auto)
    assert best_expensive["threshold"] >= best_cheap["threshold"]
