"""Orchestrates the full evaluation: agent + baselines on the golden set,
metrics, judge scoring, risk-coverage, and writes artifacts/results.json —
the single source of every number quoted in REPORT.md.

Two modes:
  --live    calls the real APIs (generator + judge), populating the cache
  (default) replays from artifacts/llm_cache.jsonl only, no network, no keys

`make reproduce` runs the default mode and asserts the recomputed numbers
match REPORT.md — see IMPLEMENTATION_PLAN.md §11.

This module remains an unfinished integration point. AI-assigned reference
labels now live in data/labels/golden_ai_v1.jsonl; the loader preserves their
provenance. Their existence does not establish human evaluation or make the
generation/judging loop complete.
"""

from __future__ import annotations

import argparse
import hashlib
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
    ai_path = ROOT / "data/labels/golden_ai_v1.jsonl"
    source = GOLDEN_PATH if GOLDEN_PATH.exists() else ai_path
    if not source.exists():
        raise FileNotFoundError(
            "No reference labels exist. Use `make label-ai` for explicitly AI-assigned "
            "labels or `make label` for an actual human annotation session."
        )
    records = _load_jsonl(source)
    if source == ai_path:
        manifest_path = ai_path.parent / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("AI label run is incomplete: no manifest")
        expected = json.loads(manifest_path.read_text())["sets"]["golden"]
        if len(records) != expected["count"] or hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("AI label file does not match its completed manifest")
        if any(r.get("label_source") != "ai" or r.get("human_reviewed") is not False for r in records):
            raise ValueError("AI reference labels have invalid provenance")
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
