"""Judge validation study: per-dimension human agreement + decoy controls.

Reads data/golden/reply_ratings.jsonl (human ratings
--reply-rating) and the matching judge scores in artifacts/judge_scores.jsonl,
and reports, per dimension:

  - Spearman rho (rank agreement)
  - quadratic-weighted Cohen's kappa (agreement accounting for chance,
    penalising larger disagreements more)

against three scorers: the real judge, the length-only decoy, and the random
decoy. If the real judge doesn't clearly beat length-only, that's reported as
the honest result, not smoothed over. Also reports a position-bias flip rate
from the evidence-order-swap re-judging pass.

Reference provenance determines the interpretation: AI-rated references measure
AI–AI agreement, and only actual human ratings measure judge–human agreement.
Agreement is a reliability diagnostic, not a mathematical ceiling on quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
RATINGS_PATH = ROOT / "data/golden/reply_ratings.jsonl"
JUDGE_SCORES_PATH = ROOT / "artifacts/judge_scores.jsonl"
SWAPPED_SCORES_PATH = ROOT / "artifacts/judge_scores_swapped.jsonl"

DIMENSIONS = ["groundedness", "resolution_helpfulness", "brand_voice", "safety"]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, ties handled via average rank. No scipy
    dependency — implemented directly so this module has no extra deps."""
    def rank(values: list[float]) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty(len(values))
        ranks[order] = np.arange(1, len(values) + 1)
        # average ranks for ties
        values_arr = np.asarray(values)
        for v in set(values):
            mask = values_arr == v
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    if len(x) < 2:
        return float("nan")
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def quadratic_weighted_kappa(x: list[int], y: list[int], min_rating: int = 1, max_rating: int = 5) -> float:
    n_ratings = max_rating - min_rating + 1
    observed = np.zeros((n_ratings, n_ratings))
    for a, b in zip(x, y):
        observed[a - min_rating, b - min_rating] += 1
    n = observed.sum()
    if n == 0:
        return float("nan")
    weights = np.zeros((n_ratings, n_ratings))
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i, j] = (i - j) ** 2 / (n_ratings - 1) ** 2

    hist_x = observed.sum(axis=1)
    hist_y = observed.sum(axis=0)
    expected = np.outer(hist_x, hist_y) / n

    observed_weighted = (weights * observed).sum()
    expected_weighted = (weights * expected).sum()
    if expected_weighted == 0:
        return float("nan")
    return float(1 - observed_weighted / expected_weighted)


def _match_key(item_id: str, system: str) -> tuple[str, str]:
    """Rating-pool item_ids for the simple baseline carry a "_simple" suffix
    (added in run_eval.py so agent/simple rows sharing a golden id stay
    unique in the flat rating CSV); judge_scores.jsonl never adds it. Both
    files do carry `system`, so match on (base_id, system) rather than the
    raw item_id — matching on item_id alone silently collapses the two
    systems' judge scores for a shared id (last-write-wins) and drops every
    simple-baseline row as an unmatched id."""
    base_id = item_id[: -len("_simple")] if item_id.endswith("_simple") else item_id
    return (base_id, system)


def agreement_report(human: list[dict], scorer: list[dict], scorer_name: str) -> dict:
    """human, scorer: lists of {"item_id": ..., "system": ..., "scores": {dim: 1-5}},
    aligned by (item_id, system) rather than item_id alone — see _match_key."""
    scorer_by_key = {_match_key(s["item_id"], s.get("system", "")): s["scores"] for s in scorer}
    human_by_key = {_match_key(h["item_id"], h.get("system", "")): h["scores"] for h in human}
    report: dict = {"scorer": scorer_name, "n_items": 0, "per_dimension": {}}
    sources = {r.get("rating_source", r.get("label_source", "unknown")) for r in human}
    report["reference_source"] = next(iter(sources)) if len(sources) == 1 else "mixed_or_missing"
    report["agreement_type"] = {
        "human": "judge–human agreement", "ai": "AI–AI agreement"
    }.get(report["reference_source"], "unverified-reference agreement")
    paired_keys = [k for k in human_by_key if k in scorer_by_key]
    report["n_items"] = len(paired_keys)
    for dim in DIMENSIONS:
        h_vals = [human_by_key[k][dim] for k in paired_keys]
        s_vals = [scorer_by_key[k][dim] for k in paired_keys]
        if not h_vals:
            continue
        report["per_dimension"][dim] = {
            "spearman_rho": round(spearman_rho(h_vals, s_vals), 3),
            "quadratic_weighted_kappa": round(quadratic_weighted_kappa(h_vals, s_vals), 3),
            "n": len(h_vals),
        }
    return report


def position_bias_flip_rate(original: list[dict], swapped: list[dict]) -> float:
    """Fraction of items where any dimension's score changed after swapping
    evidence order in the judge prompt.

    `original` is judge_scores.jsonl, which holds one row per (item, system)
    sharing the bare item_id (agent and simple_bm25_copy both score against
    the same golden id). The re-judged `swapped` set is agent-only, so the
    lookup below must filter to system == "agent" before collapsing to a
    plain id dict — otherwise the agent's swapped score gets compared
    against the *simple baseline's* original score for that id (last write
    wins), which isn't a reorder of the same reply at all."""
    orig_by_id = {o["item_id"]: o["scores"] for o in original if o.get("system") == "agent"}
    flips = 0
    n = 0
    for item in swapped:
        base = orig_by_id.get(item["item_id"])
        if base is None:
            continue
        n += 1
        if any(base[d] != item["scores"][d] for d in DIMENSIONS):
            flips += 1
    return flips / n if n else float("nan")


def run(out_path: Path = ROOT / "artifacts/judge_agreement_report.json") -> dict:
    human = _load_jsonl(RATINGS_PATH)
    if not human:
        raise ValueError("No reply ratings exist. Message labels cannot substitute for reply-quality ratings.")
    judge_scores = _load_jsonl(JUDGE_SCORES_PATH)
    swapped = _load_jsonl(SWAPPED_SCORES_PATH)
    reply_pool = {r["item_id"]: r["reply_draft"] for r in _load_jsonl(ROOT / "artifacts/predictions_for_rating.jsonl")}

    from src.eval.judge import length_only_judge, random_judge

    length_scores = [
        {"item_id": h["item_id"], "scores": {d: length_only_judge(reply_pool.get(h["item_id"], "")).__dict__[d] for d in DIMENSIONS}}
        for h in human
    ]
    random_scores = [
        {"item_id": h["item_id"], "scores": {d: random_judge(h["item_id"]).__dict__[d] for d in DIMENSIONS}}
        for h in human
    ]

    report = {
        "real_judge": agreement_report(human, judge_scores, "gemini-2.5-pro"),
        "length_decoy": agreement_report(human, length_scores, "length_only_decoy"),
        "random_decoy": agreement_report(human, random_scores, "random_decoy"),
        "position_bias_flip_rate": position_bias_flip_rate(judge_scores, swapped),
        "note": "Interpret agreement according to reference_source. AI references are not human validation.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
