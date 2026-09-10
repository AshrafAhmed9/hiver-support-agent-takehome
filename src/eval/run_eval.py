"""Orchestrates the full evaluation: agent + baselines on the golden set,
metrics, judge scoring, risk-coverage, and writes artifacts/results.json —
the single source of every number quoted in REPORT.md.

Two modes:
  --live    calls the real APIs (generator + judge), populating the cache
  (default) replays from artifacts/llm_cache.jsonl only, no network, no keys

`make reproduce` runs the default mode and asserts the recomputed numbers
match REPORT.md — see IMPLEMENTATION_PLAN.md §11.

This module requires data/golden/golden_v1.jsonl to exist and be complete
(200 items: 150 bulk-accepted after CSV review + 50 independently labelled
blind, per DECISIONS.md). It is not runnable to completion until that
labelling session finishes — see IMPLEMENTATION_PLAN.md §7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines import TfidfIntentClassifier
from src.config import BRAND

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "data/golden/golden_v1.jsonl"
CORPUS_PATH = ROOT / "data/interim" / f"{BRAND}_train.jsonl"
DEV_PATH = ROOT / "data/interim" / f"{BRAND}_dev.jsonl"
RESULTS_PATH = ROOT / "artifacts/results.json"
PREDICTIONS_PATH = ROOT / "artifacts/predictions_agent.jsonl"
JUDGE_SCORES_PATH = ROOT / "artifacts/judge_scores.jsonl"
RATING_POOL_PATH = ROOT / "artifacts/predictions_for_rating.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} does not exist yet. Run `make accept-suggested` and "
            "`make label-blind-only` first — see IMPLEMENTATION_PLAN.md §7."
        )
    records = _load_jsonl(GOLDEN_PATH)
    if len(records) < 200:
        raise ValueError(
            f"Only {len(records)}/200 golden labels exist. Run `make label-blind-only` "
            "to complete the remaining blind items before evaluating."
        )
    return records


def fit_weak_intent_classifier(train_labels: list[dict]) -> TfidfIntentClassifier:
    """Fits the TF-IDF baseline classifier on weakly-labelled train/dev data
    (pre-annotator suggestions), never on the golden set. See DECISIONS.md."""
    texts = [r["customer_text"] for r in train_labels]
    intents = [r["intent"] for r in train_labels]
    return TfidfIntentClassifier().fit(texts, intents)


def run(live: bool = False) -> dict:
    # Do not initialize providers or imply evaluation is blocked on human
    # labels: the integration loop itself is not implemented yet.
    load_golden()
    raise NotImplementedError(
        "Reference labels are available, but the end-to-end generation/judging "
        "loop still needs implementation. AI labels do not supply human agreement."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    results = run(live=args.live)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
