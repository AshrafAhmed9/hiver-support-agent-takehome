"""Runs the recalibrated judge (judge_v2) on the 60-item holdout split and
re-measures agreement against those same 60 human ratings, so it's a fair
comparison against the original judge's numbers on the same items.

Usage:
    uv run python -m src.eval.run_judge_v2 --live   # calls the real judge
    uv run python -m src.eval.run_judge_v2           # replays from cache

Writes artifacts/judge_scores_v2_holdout.jsonl and
artifacts/judge_agreement_v2_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import BRAND, JUDGE_MODEL
from src.eval.judge_agreement import agreement_report
from src.eval.judge_v2 import build_rubric_v2, calibration_holdout_split, judge_reply_v2
from src.llm import CachedLLM, groq_call_fn
from src.retrieve import BM25Retriever, load_historical_pairs

ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = ROOT / "data/golden/reply_ratings.jsonl"
POOL_PATH = ROOT / "artifacts/predictions_for_rating.jsonl"
CORPUS_PATH = ROOT / "data/interim" / f"{BRAND}_train.jsonl"
V2_SCORES_PATH = ROOT / "artifacts/judge_scores_v2_holdout.jsonl"
V2_REPORT_PATH = ROOT / "artifacts/judge_agreement_v2_report.json"
ORIGINAL_REPORT_PATH = ROOT / "artifacts/judge_agreement_report.json"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def run(live: bool = False) -> dict:
    human = _load_jsonl(RATINGS_PATH)
    pool = {p["item_id"]: p for p in _load_jsonl(POOL_PATH)}
    _, holdout_ids = calibration_holdout_split(human)
    holdout_set = set(holdout_ids)
    holdout_ratings = [r for r in human if r["item_id"] in holdout_set]

    rubric_v2 = build_rubric_v2()
    llm = CachedLLM(JUDGE_MODEL, groq_call_fn(JUDGE_MODEL), allow_live=live)

    # Reconstruct evidence the same way run_eval.py judged these replies
    # originally: for the simple_bm25_copy baseline, its own reply IS the
    # evidence (it's a verbatim historical reply); for the agent, retrieve
    # fresh against the same corpus. This isn't byte-identical to the exact
    # evidence spans the agent cited at generation time (that would require
    # re-running the agent's LLM call, which this recalibration test doesn't
    # need), but it's the same retrieval the agent draws from, not a
    # no-evidence judgment, so groundedness stays a fair comparison.
    corpus = load_historical_pairs(CORPUS_PATH)
    retriever = BM25Retriever(corpus)

    # Same fallback pattern as run_eval.py's _judge_or_flag: one malformed
    # response must not kill a 60-item batch. Falls back to a neutral score
    # and records the item as flagged rather than crashing or silently
    # treating the fallback as a real judgment.
    judge_failures: list[dict] = []
    v2_scores = []
    for r in holdout_ratings:
        item = pool[r["item_id"]]
        if r["system"] == "simple_bm25_copy":
            evidence_texts = [item["reply_draft"]]
        else:
            retrieved = retriever.search(item["customer_text"], limit=5)
            evidence_texts = [e.brand_reply for e in retrieved] or [item["customer_text"]]
        try:
            score = judge_reply_v2(llm, rubric_v2, item["customer_text"], evidence_texts, item["reply_draft"])
        except ValueError as exc:
            from src.eval.judge import JudgeScore

            judge_failures.append({"item_id": r["item_id"], "error": str(exc)})
            score = JudgeScore(3, 3, 3, 3, rationale=f"FALLBACK: judge_v2 parsing failed after retries ({exc})")
        v2_scores.append({"item_id": r["item_id"], "system": r["system"], "scores": score.__dict__})

    V2_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with V2_SCORES_PATH.open("w", encoding="utf-8") as f:
        for s in v2_scores:
            f.write(json.dumps(s) + "\n")

    v2_report = agreement_report(holdout_ratings, v2_scores, f"{JUDGE_MODEL}-v2-fewshot")
    v2_report["n_calibration_examples_shown"] = 20
    v2_report["holdout_size"] = len(holdout_ratings)

    # Same 60 items, scored by the original zero-shot judge, for a fair
    # apples-to-apples before/after (not the full-80 number from
    # judge_agreement_report.json, which includes the 20 calibration items).
    original_judge_scores = [s for s in _load_jsonl(ROOT / "artifacts/judge_scores.jsonl")]
    v1_report_on_holdout = agreement_report(holdout_ratings, original_judge_scores, JUDGE_MODEL)

    comparison = {
        "holdout_size": len(holdout_ratings),
        "v1_zero_shot_on_holdout": v1_report_on_holdout,
        "v2_fewshot_on_holdout": v2_report,
        "judge_v2_parse_failures": judge_failures,
    }
    V2_REPORT_PATH.write_text(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(live=args.live), indent=2))
