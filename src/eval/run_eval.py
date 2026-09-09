"""Orchestrates the full evaluation: agent + baselines on the golden set,
metrics, judge scoring, risk-coverage, and writes artifacts/results.json —
the single source of every number quoted in REPORT.md.

Two modes:
  --live    calls the real APIs (generator + judge), populating the cache
  (default) replays from artifacts/llm_cache.jsonl only, no network, no keys

`make reproduce` runs the default mode and asserts the recomputed numbers
match REPORT.md — see IMPLEMENTATION_PLAN.md §11.

This module requires data/golden/golden_v1.jsonl to exist (produced by a
human running src/label_tui.py — see IMPLEMENTATION_PLAN.md §7). It is not
runnable to completion until that labelling session happens; every piece
that doesn't depend on it (baseline fitting, retrieval, judge/decoy wiring)
is exercised by tests/test_run_eval_pipeline.py against synthetic golden data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines import TfidfIntentClassifier
from src.config import BRAND, GENERATOR_MODEL
from src.eval.judge import make_judge_llm
from src.llm import CachedLLM, groq_call_fn
from src.retrieve import BM25Retriever, load_historical_pairs

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
            f"{GOLDEN_PATH} does not exist yet. This requires a human labelling "
            "session (uv run python -m src.label_tui) — see IMPLEMENTATION_PLAN.md §7."
        )
    return _load_jsonl(GOLDEN_PATH)


def fit_weak_intent_classifier(train_labels: list[dict]) -> TfidfIntentClassifier:
    """Fits the TF-IDF baseline classifier on weakly-labelled train/dev data
    (pre-annotator suggestions), never on the golden set. See DECISIONS.md."""
    texts = [r["customer_text"] for r in train_labels]
    intents = [r["intent"] for r in train_labels]
    return TfidfIntentClassifier().fit(texts, intents)


def run(live: bool = False) -> dict:
    golden = load_golden()
    corpus = load_historical_pairs(CORPUS_PATH)
    retriever = BM25Retriever(corpus)

    generator = CachedLLM(GENERATOR_MODEL, groq_call_fn(GENERATOR_MODEL), allow_live=live)
    judge_llm = make_judge_llm(allow_live=live)

    # ... generation over `golden`, judge scoring, metric computation would
    # run here in the full build. Left as the integration point: everything
    # it depends on (retriever, generator, judge, metrics, risk-coverage) is
    # independently unit-tested. Wiring this end-to-end is the next step
    # once data/golden/golden_v1.jsonl exists.
    raise NotImplementedError(
        "End-to-end generation+judging loop is the next implementation step, "
        "blocked on the golden-set labelling session (§7). All dependencies "
        "(retriever, generator, judge, decoys, metrics, risk-coverage) are "
        "implemented and unit-tested independently."
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
