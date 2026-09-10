"""Orchestrates the full evaluation: agent + baselines on the golden set,
metrics, judge scoring, risk-coverage, and writes artifacts/results.json —
the single source of every number quoted in REPORT.md.

Two modes:
  --live    calls the real APIs (generator + judge), populating the cache
  (default) replays from artifacts/llm_cache.jsonl only, no network, no keys

`make reproduce` runs the default mode and asserts the recomputed numbers
match REPORT.md — see IMPLEMENTATION_PLAN.md §11.

This module requires data/golden/golden_v1.jsonl to exist and be complete
(250 items — see REPORT.md's golden-set section for how they were produced).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.agent import make_groq_generator, run_agent
from src.baselines import BM25CopyBaseline, TfidfIntentClassifier, TrivialBaseline
from src.config import BRAND, COST_RATIO_SWEEP, RANDOM_SEED, TARGET_SAFE_AUTO_REPLY_RATE
from src.eval.judge import (
    judge_reply,
    make_judge_llm,
    swap_order_prompt,
)
from src.eval.metrics import (
    accuracy_ci,
    confusion_matrix,
    macro_f1,
    routing_precision_recall_f1,
)
from src.eval.risk_coverage import (
    GoldenItem,
    cost_ratio_sensitivity,
    coverage_at_safety_target,
    risk_coverage_curve,
)
from src.retrieve import BM25Retriever, load_historical_pairs
from src.sampling import fit_pseudo_labeller, pseudo_label

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "data/golden/golden_v1.jsonl"
CORPUS_PATH = ROOT / "data/interim" / f"{BRAND}_train.jsonl"
RESULTS_PATH = ROOT / "artifacts/results.json"
PREDICTIONS_AGENT_PATH = ROOT / "artifacts/predictions_agent.jsonl"
PREDICTIONS_SIMPLE_PATH = ROOT / "artifacts/predictions_simple.jsonl"
PREDICTIONS_TRIVIAL_PATH = ROOT / "artifacts/predictions_trivial.jsonl"
JUDGE_SCORES_PATH = ROOT / "artifacts/judge_scores.jsonl"
JUDGE_SCORES_SWAPPED_PATH = ROOT / "artifacts/judge_scores_swapped.jsonl"
RATING_POOL_PATH = ROOT / "artifacts/predictions_for_rating.jsonl"

N_POSITION_BIAS_SAMPLE = 30
N_RATING_POOL = 80


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"{GOLDEN_PATH} does not exist yet — see IMPLEMENTATION_PLAN.md §7.")
    records = _load_jsonl(GOLDEN_PATH)
    if len(records) < 250:
        raise ValueError(f"Only {len(records)}/250 golden labels exist; complete the remaining items before evaluating.")
    return records


def fit_weak_intent_classifier(corpus_path: Path, sample_size: int = 3000, seed: int = RANDOM_SEED) -> TfidfIntentClassifier:
    """Fits the TF-IDF baseline classifier on weakly-labelled train-split
    data. Labels come from the non-LLM nearest-centroid pseudo-labeller in
    src/sampling.py — no LLM calls, and disjoint from the golden set (drawn
    from train, golden is drawn from test_pool). See DECISIONS.md."""
    records = _load_jsonl(corpus_path)
    texts_all = [r["customer_text"] for r in records if r.get("customer_text")]
    rng = random.Random(seed)
    sample = texts_all if len(texts_all) <= sample_size else rng.sample(texts_all, k=sample_size)

    vectorizer, centroids, intents = fit_pseudo_labeller()
    weak_labels = [pseudo_label(vectorizer, centroids, intents, t)[0] for t in sample]
    return TfidfIntentClassifier().fit(sample, weak_labels), weak_labels


def _judge_or_flag(judge_llm, text: str, evidence_texts: list[str], reply: str, item_id: str, system: str):
    """Judges a reply; on repeated malformed-JSON failure, falls back to a
    neutral score (3s) rather than crashing a 250-item batch job over one
    bad response, and records the item as flagged rather than silently
    treating the fallback as a real judgment."""
    try:
        return judge_reply(judge_llm, text, evidence_texts, reply)
    except ValueError as exc:
        from src.eval.judge import JudgeScore

        JUDGE_FAILURES.append({"item_id": item_id, "system": system, "error": str(exc)})
        return JudgeScore(3, 3, 3, 3, rationale=f"FALLBACK: judge parsing failed after retries ({exc})")


JUDGE_FAILURES: list[dict] = []


def run(live: bool = False, sample_size: int | None = None) -> dict:
    """sample_size: if given, evaluates the first N golden items (in golden_v1.jsonl
    order, which is already a one-time-seeded shuffle from src/sampling.py —
    not cherry-picked). The brief explicitly allows this: "we will not run
    your code on the full dataset — a subsample is expected and encouraged."
    Used in practice because the generator model's Groq daily token quota
    (200k TPD) doesn't stretch to 250 live generations in one sitting — see
    DECISIONS.md and REPORT.md's "misleading headline number" section."""
    JUDGE_FAILURES.clear()
    golden = load_golden()
    if sample_size is not None:
        golden = golden[:sample_size]
    corpus = load_historical_pairs(CORPUS_PATH)
    retriever = BM25Retriever(corpus)

    clf, weak_train_intents = fit_weak_intent_classifier(CORPUS_PATH)
    generate = make_groq_generator(allow_live=live)
    judge_llm = make_judge_llm(allow_live=live)

    trivial = TrivialBaseline(weak_train_intents, route="escalate")
    bm25_copy = BM25CopyBaseline(retriever)

    agent_predictions, simple_predictions, trivial_predictions = [], [], []
    judge_scores = []
    golden_items_for_curve: list[GoldenItem] = []

    trivial_score = None  # judged once, reused (fixed canned reply)

    for item in golden:
        text = item["customer_text"]
        gold_escalate = item["should_escalate"]

        # --- agent ---
        result = run_agent(text, retriever, generate)
        agent_predictions.append(
            {
                "item_id": item["id"],
                "intent": result.intent,
                "reply_draft": result.reply_draft,
                "route": result.route,
                "reason_codes": list(result.reason_codes),
            }
        )
        evidence_texts = [c.supporting_span for c in result.evidence] or [text]
        score = _judge_or_flag(judge_llm, text, evidence_texts, result.reply_draft, item["id"], "agent")
        judge_scores.append({"item_id": item["id"], "system": "agent", "scores": score.__dict__})

        clf_proba = clf.model.predict_proba(clf.vectorizer.transform([text]))[0]
        clf_max_conf = float(clf_proba.max())
        # Guardrail is a hard gate (DECISIONS.md #8): anything it escalates
        # can never be auto-handled regardless of threshold. Confidence for
        # the coverage sweep only applies within the guardrail-passed pool.
        confidence = clf_max_conf if result.route == "auto" else 0.0
        is_safe = (not gold_escalate) and (score.mean() >= 4.0)
        golden_items_for_curve.append(GoldenItem(confidence=confidence, is_safe_if_auto=is_safe))

        # --- simple baseline: BM25-copy reply, TF-IDF classifier intent ---
        simple_intent = clf.predict([text])[0]
        bm25_pred = bm25_copy.predict(text)
        simple_predictions.append(
            {
                "item_id": item["id"],
                "intent": simple_intent,
                "reply_draft": bm25_pred.reply_draft,
                "route": bm25_pred.route,
            }
        )
        simple_score = _judge_or_flag(judge_llm, text, [bm25_pred.reply_draft], bm25_pred.reply_draft, item["id"], "simple_bm25_copy")
        judge_scores.append({"item_id": item["id"], "system": "simple_bm25_copy", "scores": simple_score.__dict__})

        # --- trivial baseline: majority intent, fixed canned reply ---
        trivial_pred = trivial.predict(text)
        trivial_predictions.append(
            {"item_id": item["id"], "intent": trivial_pred.intent, "reply_draft": trivial_pred.reply_draft, "route": trivial_pred.route}
        )
        if trivial_score is None:
            trivial_score = judge_reply(judge_llm, text, [], trivial_pred.reply_draft)

    judge_scores.append({"item_id": "_trivial_canned_reply", "system": "trivial", "scores": trivial_score.__dict__})

    _write_jsonl(agent_predictions, PREDICTIONS_AGENT_PATH)
    _write_jsonl(simple_predictions, PREDICTIONS_SIMPLE_PATH)
    _write_jsonl(trivial_predictions, PREDICTIONS_TRIVIAL_PATH)
    _write_jsonl(judge_scores, JUDGE_SCORES_PATH)

    # --- position-bias check: re-judge a subset with evidence order swapped ---
    rng = random.Random(RANDOM_SEED)
    subset_ids = set(rng.sample([g["id"] for g in golden], k=min(N_POSITION_BIAS_SAMPLE, len(golden))))
    swapped_scores = []
    agent_pred_by_id = {p["item_id"]: p for p in agent_predictions}
    golden_by_id = {g["id"]: g for g in golden}
    for item_id in subset_ids:
        pred = agent_pred_by_id[item_id]
        gold = golden_by_id[item_id]
        retrieved = retriever.search(gold["customer_text"], limit=5)
        evidence_texts = [e.brand_reply for e in retrieved] or [gold["customer_text"]]
        prompt = swap_order_prompt(gold["customer_text"], evidence_texts, pred["reply_draft"])
        response = judge_llm.generate(prompt, {"temperature": 0.0})
        import re as _re

        match = _re.search(r"\{.*\}", response.text, _re.DOTALL)
        data = json.loads(match.group())
        swapped_scores.append(
            {
                "item_id": item_id,
                "scores": {
                    "groundedness": int(data["groundedness"]),
                    "resolution_helpfulness": int(data["resolution_helpfulness"]),
                    "brand_voice": int(data["brand_voice"]),
                    "safety": int(data["safety"]),
                },
            }
        )
    _write_jsonl(swapped_scores, JUDGE_SCORES_SWAPPED_PATH)

    # --- rating pool for human reply-quality ratings ---
    rating_rng = random.Random(RANDOM_SEED)
    n_each = min(N_RATING_POOL // 2, len(agent_predictions), len(simple_predictions))
    agent_sample_ids = rating_rng.sample([p["item_id"] for p in agent_predictions], k=n_each)
    simple_sample_ids = rating_rng.sample([p["item_id"] for p in simple_predictions], k=n_each)
    rating_pool = []
    for pred in agent_predictions:
        if pred["item_id"] in agent_sample_ids:
            gold = golden_by_id[pred["item_id"]]
            rating_pool.append({"item_id": pred["item_id"], "system": "agent", "customer_text": gold["customer_text"], "reply_draft": pred["reply_draft"]})
    for pred in simple_predictions:
        if pred["item_id"] in simple_sample_ids:
            gold = golden_by_id[pred["item_id"]]
            rating_pool.append({"item_id": pred["item_id"] + "_simple", "system": "simple_bm25_copy", "customer_text": gold["customer_text"], "reply_draft": pred["reply_draft"]})
    rating_rng.shuffle(rating_pool)
    _write_jsonl(rating_pool, RATING_POOL_PATH)

    # --- metrics ---
    gold_intents = [g["intent"] for g in golden]
    gold_escalate = [g["should_escalate"] for g in golden]
    agent_intents = [p["intent"] for p in agent_predictions]
    simple_intents = [p["intent"] for p in simple_predictions]
    trivial_intents = [p["intent"] for p in trivial_predictions]
    agent_route_escalate = [p["route"] == "escalate" for p in agent_predictions]
    simple_route_escalate = [p["route"] == "escalate" for p in simple_predictions]
    trivial_escalate_all_route = [True] * len(golden)
    trivial_auto_all_route = [False] * len(golden)

    agent_acc, agent_acc_lo, agent_acc_hi = accuracy_ci(gold_intents, agent_intents)
    simple_acc, simple_acc_lo, simple_acc_hi = accuracy_ci(gold_intents, simple_intents)
    trivial_acc, trivial_acc_lo, trivial_acc_hi = accuracy_ci(gold_intents, trivial_intents)

    curve = risk_coverage_curve(golden_items_for_curve)
    headline = coverage_at_safety_target(curve, target=TARGET_SAFE_AUTO_REPLY_RATE)
    sensitivity = cost_ratio_sensitivity(golden_items_for_curve, ratios=COST_RATIO_SWEEP)

    agent_reply_scores = [s["scores"] for s in judge_scores if s["system"] == "agent"]
    simple_reply_scores = [s["scores"] for s in judge_scores if s["system"] == "simple_bm25_copy"]

    def _mean_dims(scores: list[dict]) -> dict:
        dims = ["groundedness", "resolution_helpfulness", "brand_voice", "safety"]
        return {d: round(sum(s[d] for s in scores) / len(scores), 2) for d in dims} if scores else {}

    results = {
        "n_golden": len(golden),
        "intent_accuracy": {
            "agent": {"point": round(agent_acc, 3), "ci95": [round(agent_acc_lo, 3), round(agent_acc_hi, 3)]},
            "simple_baseline_tfidf": {"point": round(simple_acc, 3), "ci95": [round(simple_acc_lo, 3), round(simple_acc_hi, 3)]},
            "trivial_baseline_majority": {"point": round(trivial_acc, 3), "ci95": [round(trivial_acc_lo, 3), round(trivial_acc_hi, 3)]},
        },
        "intent_macro_f1": {
            "agent": round(macro_f1(gold_intents, agent_intents), 3),
            "simple_baseline_tfidf": round(macro_f1(gold_intents, simple_intents), 3),
            "trivial_baseline_majority": round(macro_f1(gold_intents, trivial_intents), 3),
        },
        "agent_confusion_matrix": confusion_matrix(gold_intents, agent_intents),
        "routing": {
            "agent": routing_precision_recall_f1(gold_escalate, agent_route_escalate),
            "simple_baseline_confidence_route": routing_precision_recall_f1(gold_escalate, simple_route_escalate),
            "trivial_escalate_all": routing_precision_recall_f1(gold_escalate, trivial_escalate_all_route),
            "trivial_auto_all": routing_precision_recall_f1(gold_escalate, trivial_auto_all_route),
        },
        "reply_quality_judge_means": {
            "agent": _mean_dims(agent_reply_scores),
            "simple_baseline_bm25_copy": _mean_dims(simple_reply_scores),
            "trivial_baseline_canned": trivial_score.__dict__ if trivial_score else {},
        },
        "headline_coverage_at_safety_target": {
            "target_safe_auto_reply_rate": TARGET_SAFE_AUTO_REPLY_RATE,
            **headline,
        },
        "cost_ratio_sensitivity": sensitivity,
        "position_bias_sample_size": len(swapped_scores),
        "rating_pool_size": len(rating_pool),
        "judge_parse_failures": list(JUDGE_FAILURES),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None, help="Evaluate only the first N golden items.")
    args = parser.parse_args()
    results = run(live=args.live, sample_size=args.sample_size)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
