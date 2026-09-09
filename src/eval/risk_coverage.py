"""The headline number: coverage at a fixed safe-auto-reply rate.

Not accuracy. This is what a support org actually buys — how much volume can
be auto-handled, and at what safety level. Also derives the escalation
threshold from an explicit, arguable cost model rather than a hand-picked
number, and reports how the headline moves as that cost ratio changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    COST_BAD_AUTO_REPLY,
    COST_HUMAN_TOUCH,
    COST_RATIO_SWEEP,
    TARGET_SAFE_AUTO_REPLY_RATE,
)


@dataclass(frozen=True)
class GoldenItem:
    confidence: float          # combined confidence score in [0, 1]
    is_safe_if_auto: bool      # ground truth: would auto-handling this be acceptable?
                                 # (should_escalate is False AND reply-quality judge passed)


def risk_coverage_curve(items: list[GoldenItem], n_thresholds: int = 101) -> list[dict]:
    """Sweeps the confidence threshold from 0 to 1. At each threshold t,
    'auto' = items with confidence >= t. Returns coverage and safe-rate at
    each point."""
    thresholds = np.linspace(0, 1, n_thresholds)
    curve = []
    for t in thresholds:
        auto_items = [item for item in items if item.confidence >= t]
        coverage = len(auto_items) / len(items) if items else 0.0
        safe_rate = (
            sum(item.is_safe_if_auto for item in auto_items) / len(auto_items)
            if auto_items
            else 1.0  # vacuously safe: nothing was auto-handled
        )
        curve.append({"threshold": float(t), "coverage": coverage, "safe_auto_reply_rate": safe_rate})
    return curve


def coverage_at_safety_target(curve: list[dict], target: float = TARGET_SAFE_AUTO_REPLY_RATE) -> dict:
    """Finds the highest-coverage point on the curve that still meets the
    safety target. This is the headline number."""
    eligible = [p for p in curve if p["safe_auto_reply_rate"] >= target]
    if not eligible:
        return {"threshold": 1.0, "coverage": 0.0, "safe_auto_reply_rate": 1.0, "note": "no threshold meets the target"}
    return max(eligible, key=lambda p: p["coverage"])


def cost_optimal_threshold(
    items: list[GoldenItem],
    n_thresholds: int = 101,
    cost_bad_auto: float = COST_BAD_AUTO_REPLY,
    cost_human: float = COST_HUMAN_TOUCH,
) -> dict:
    """Picks the threshold minimising expected cost:
      auto-handled & unsafe -> cost_bad_auto
      escalated              -> cost_human
      auto-handled & safe    -> 0
    This is the threshold actually used to derive the route decision, not the
    safety-target threshold above (which is what's reported as the headline).
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best = None
    for t in thresholds:
        cost = 0.0
        for item in items:
            auto = item.confidence >= t
            if auto and not item.is_safe_if_auto:
                cost += cost_bad_auto
            elif not auto:
                cost += cost_human
        if best is None or cost < best["cost"]:
            best = {"threshold": float(t), "cost": cost}
    return best


def cost_ratio_sensitivity(items: list[GoldenItem], ratios: list[float] = COST_RATIO_SWEEP) -> list[dict]:
    """How the headline coverage number moves as the assumed cost ratio
    (bad-auto-reply cost / human-touch cost) varies. Include this whenever
    the headline is quoted — a reviewer will not agree with a single fixed
    ratio, and showing the sensitivity pre-empts that objection."""
    results = []
    for ratio in ratios:
        best = cost_optimal_threshold(items, cost_bad_auto=ratio, cost_human=1.0)
        auto_items = [i for i in items if i.confidence >= best["threshold"]]
        coverage = len(auto_items) / len(items) if items else 0.0
        safe_rate = sum(i.is_safe_if_auto for i in auto_items) / len(auto_items) if auto_items else 1.0
        results.append({"cost_ratio": ratio, "threshold": best["threshold"], "coverage": coverage, "safe_auto_reply_rate": safe_rate})
    return results
