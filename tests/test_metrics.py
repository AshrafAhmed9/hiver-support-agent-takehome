from src.eval.metrics import (
    accuracy,
    accuracy_ci,
    macro_f1,
    routing_confusion,
    routing_precision_recall_f1,
)


def test_accuracy_hand_computed():
    y_true = ["a", "b", "a", "c"]
    y_pred = ["a", "b", "b", "c"]
    assert accuracy(y_true, y_pred) == 0.75


def test_macro_f1_perfect():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "a", "b"]
    assert macro_f1(y_true, y_pred) == 1.0


def test_macro_f1_hand_computed():
    # class a: tp=1 fp=1 fn=0 -> P=0.5 R=1.0 F1=0.667
    # class b: tp=1 fp=0 fn=1 -> P=1.0 R=0.5 F1=0.667
    y_true = ["a", "b", "b"]
    y_pred = ["a", "a", "b"]
    assert abs(macro_f1(y_true, y_pred) - 0.6667) < 0.001


def test_accuracy_ci_bounds_point_estimate():
    y_true = ["a"] * 8 + ["b"] * 2
    y_pred = ["a"] * 7 + ["b"] * 3  # 9/10 correct where labels align positionally... just check point matches accuracy()
    point, lo, hi = accuracy_ci(y_true, y_pred, n_resamples=500)
    assert abs(point - accuracy(y_true, y_pred)) < 1e-9
    assert lo <= point <= hi


def test_routing_confusion_business_terms():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]
    counts = routing_confusion(y_true, y_pred)
    assert counts["correct_escalation"] == 1
    assert counts["missed_escalation"] == 1
    assert counts["correct_auto"] == 1
    assert counts["needless_escalation"] == 1


def test_routing_precision_recall_f1():
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]
    result = routing_precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
