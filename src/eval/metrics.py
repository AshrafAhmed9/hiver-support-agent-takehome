"""Intent and routing metrics with bootstrap confidence intervals.

With n=200 the point estimates alone are misleading — CIs are the point.
Pure numpy/collections, no sklearn dependency here so this module is trivial
to unit-test against hand-computed cases.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    scores: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return scores


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = per_class_f1(y_true, y_pred)
    return sum(scores.values()) / len(scores) if scores else 0.0


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def bootstrap_ci(values: list[float] | list[int], statistic_fn=None, n_resamples: int = 10_000, seed: int = 20260909, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) via percentile bootstrap.

    If statistic_fn is None, bootstraps the mean of `values` directly. Pass
    statistic_fn(indices) -> float to bootstrap a metric computed jointly over
    paired arrays (e.g. accuracy over resampled (y_true, y_pred) pairs)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    arr = np.asarray(values, dtype=float)
    point = float(arr.mean()) if statistic_fn is None else statistic_fn(list(range(n)))
    resample_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        resample_stats[i] = arr[idx].mean() if statistic_fn is None else statistic_fn(idx)
    lo, hi = np.percentile(resample_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def accuracy_ci(y_true: list[str], y_pred: list[str], n_resamples: int = 10_000, seed: int = 20260909) -> tuple[float, float, float]:
    def stat(idx: list[int]) -> float:
        return sum(y_true[i] == y_pred[i] for i in idx) / len(idx)

    return bootstrap_ci(list(range(len(y_true))), statistic_fn=stat, n_resamples=n_resamples, seed=seed)


def routing_confusion(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict[str, int]:
    """Named in business terms: missed_escalation = auto-replied when it
    shouldn't have (expensive); needless_escalation = punted an easy one
    (cheap)."""
    counts = Counter()
    for t, p in zip(y_true_escalate, y_pred_escalate):
        if t and p:
            counts["correct_escalation"] += 1
        elif not t and not p:
            counts["correct_auto"] += 1
        elif t and not p:
            counts["missed_escalation"] += 1
        else:
            counts["needless_escalation"] += 1
    return dict(counts)


def routing_precision_recall_f1(y_true_escalate: list[bool], y_pred_escalate: list[bool]) -> dict[str, float]:
    counts = routing_confusion(y_true_escalate, y_pred_escalate)
    tp = counts.get("correct_escalation", 0)
    fp = counts.get("needless_escalation", 0)
    fn = counts.get("missed_escalation", 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
