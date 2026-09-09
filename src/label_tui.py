"""Terminal labelling tool for the golden set (§7 of the plan).

Plain input(), no curses dependency, one item per screen. For non-blind
items, shows the pre-annotator's suggestion and asks accept/override. For
blind items, shows no suggestion at all. Every keystroke is logged so the
override rate and the anchoring-bias comparison (suggested vs. blind
agreement) can be computed later without re-running anything.

Usage:
    uv run python -m src.label_tui                  # intent + escalation labelling
    uv run python -m src.label_tui --resume          # continue an interrupted session
    uv run python -m src.label_tui --reply-rating    # separate pass: rate ~80 replies

This file is built so it's ready to run; it is not run automatically because
the labelling itself needs a human — see IMPLEMENTATION_PLAN.md §7.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import INTENTS

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/golden/golden_candidates.jsonl"
GOLDEN_PATH = ROOT / "data/golden/golden_v1.jsonl"
LOG_PATH = ROOT / "data/golden/labeling_log.jsonl"
RATINGS_PATH = ROOT / "data/golden/reply_ratings.jsonl"
RATING_POOL_PATH = ROOT / "artifacts" / "predictions_for_rating.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _print_item(record: dict, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"[{index + 1}/{total}]  id={record['id']}  difficulty={record.get('difficulty')}")
    context = record.get("context") or []
    if context:
        print("-- prior context --")
        for turn in context[-3:]:
            role = turn.get("author_role", "?")
            print(f"  ({role}) {turn.get('text', '')[:200]}")
    print("-- customer message --")
    print(f"  {record['customer_text']}")


def _prompt_intent(default_hint: str | None) -> str:
    print("\nIntents:")
    for i, intent in enumerate(INTENTS):
        marker = " <-- suggested" if intent == default_hint else ""
        print(f"  {i}: {intent}{marker}")
    while True:
        raw = input("Enter intent number (blank = accept suggestion): ").strip()
        if raw == "" and default_hint:
            return default_hint
        if raw.isdigit() and 0 <= int(raw) < len(INTENTS):
            return INTENTS[int(raw)]
        print("Invalid input, try again.")


def _prompt_yes_no(question: str) -> bool:
    while True:
        raw = input(f"{question} [y/n]: ").strip().lower()
        if raw in {"y", "n"}:
            return raw == "y"
        print("Enter y or n.")


def run_intent_labelling(resume: bool) -> None:
    candidates = _load_jsonl(CANDIDATES_PATH)
    if not candidates:
        print(f"No candidates found at {CANDIDATES_PATH}. Run `python -m src.sampling` first.")
        return
    already_done = {r["id"] for r in _load_jsonl(GOLDEN_PATH)} if resume else set()
    remaining = [c for c in candidates if c["id"] not in already_done]
    print(f"{len(remaining)} of {len(candidates)} items remaining.")

    for index, record in enumerate(remaining):
        _print_item(record, index, len(remaining))
        suggestion = record.get("suggested_intent")
        blind = bool(record.get("blind"))
        if blind:
            print("(no suggestion shown — blind item)")

        started = time.monotonic()
        final_intent = _prompt_intent(suggestion if not blind else None)
        should_escalate = _prompt_yes_no("Should this escalate to a human?")
        escalate_reason = input("One-line reason: ").strip()
        difficulty = record.get("difficulty", "medium")
        elapsed = round(time.monotonic() - started, 1)

        overridden = (not blind) and suggestion is not None and final_intent != suggestion
        _append_jsonl(
            LOG_PATH,
            {
                "id": record["id"],
                "suggested_intent": suggestion,
                "final_intent": final_intent,
                "overridden": overridden,
                "blind": blind,
                "seconds": elapsed,
            },
        )
        _append_jsonl(
            GOLDEN_PATH,
            {
                "id": record["id"],
                "thread_id": record["message_id"],
                "customer_text": record["customer_text"],
                "context": record.get("context", []),
                "intent": final_intent,
                "secondary_intent": None,
                "should_escalate": should_escalate,
                "escalate_reason": escalate_reason,
                "difficulty": difficulty,
                "blind": blind,
            },
        )
        print(f"Saved. ({elapsed}s)")


def run_reply_rating() -> None:
    """Second pass: rate ~80 replies on the same rubric the LLM judge uses,
    blind to which system produced each reply. Requires
    artifacts/predictions_for_rating.jsonl to exist (built by src/eval/run_eval.py
    after the agent and baselines have produced predictions)."""
    pool = _load_jsonl(RATING_POOL_PATH)
    if not pool:
        print(f"No rating pool at {RATING_POOL_PATH}. Run the eval pipeline first.")
        return
    already_done = {r["item_id"] for r in _load_jsonl(RATINGS_PATH)}
    remaining = [p for p in pool if p["item_id"] not in already_done]
    print(f"{len(remaining)} of {len(pool)} replies remaining to rate.")

    dims = ["groundedness", "resolution_helpfulness", "brand_voice", "safety"]
    anchors = "1=poor  3=acceptable  5=excellent"
    for index, item in enumerate(remaining):
        print("\n" + "=" * 72)
        print(f"[{index + 1}/{len(remaining)}] {item['item_id']}  (system hidden)")
        print(f"customer: {item['customer_text']}")
        print(f"reply: {item['reply_draft']}")
        scores = {}
        for dim in dims:
            while True:
                raw = input(f"{dim} ({anchors}): ").strip()
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    scores[dim] = int(raw)
                    break
                print("Enter 1-5.")
        _append_jsonl(
            RATINGS_PATH,
            {"item_id": item["item_id"], "system": item["system"], "scores": scores},
        )
        print("Saved.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reply-rating", action="store_true")
    args = parser.parse_args()
    if args.reply_rating:
        run_reply_rating()
    else:
        run_intent_labelling(args.resume)


if __name__ == "__main__":
    main()
