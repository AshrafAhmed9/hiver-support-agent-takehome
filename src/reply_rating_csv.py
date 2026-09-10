"""CSV export/import for the human reply-quality ratings that validate the
LLM judge (§9.5 of the plan).

Deliberately blind. This file exists for exactly one purpose: measuring
whether the LLM judge agrees with an independent human. If the rating were
pre-filled from any model output, "judge vs. human" agreement would quietly
become "judge vs. a model, confirmed by a human" — a materially weaker
claim than the evidence the brief asks for.

Workflow:
    uv run python -m src.reply_rating_csv export
    ... rate every row 1-5 on all 4 dimensions, blind, from scratch ...
    uv run python -m src.reply_rating_csv import
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "artifacts/predictions_for_rating.jsonl"
CSV_PATH = ROOT / "data/golden/reply_ratings.csv"
RATINGS_PATH = ROOT / "data/golden/reply_ratings.jsonl"

DIMENSIONS = ["groundedness", "resolution_helpfulness", "brand_voice", "safety"]
RUBRIC_HINT = (
    "1=poor 3=acceptable 5=excellent. groundedness: only proposes remedies plausibly "
    "supported by real historical guidance for this brand. resolution_helpfulness: "
    "would this move the issue forward. brand_voice: sounds like this brand's real "
    "support team, not generic. safety: no unverifiable promises (refunds/dates/account actions)."
)

FIELDNAMES = ["item_id", "customer_text", "reply_draft", *DIMENSIONS]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def export_csv() -> None:
    if not POOL_PATH.exists():
        print(f"{POOL_PATH} does not exist. Run the eval pipeline first (src.eval.run_eval).")
        return
    pool = _load_jsonl(POOL_PATH)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in pool:
            # `system` deliberately omitted from the CSV — blind rating.
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "customer_text": item["customer_text"],
                    "reply_draft": item["reply_draft"],
                    **{d: "" for d in DIMENSIONS},
                }
            )
    print(f"Wrote {len(pool)} rows to {CSV_PATH}.")
    print(RUBRIC_HINT)
    print("Fill in all 4 dimension columns (1-5) for every row, then run:")
    print("  uv run python -m src.reply_rating_csv import")


def import_csv() -> None:
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} does not exist. Run `export` first.")
        return
    pool_by_id = {p["item_id"]: p for p in _load_jsonl(POOL_PATH)}

    ratings = []
    errors = []
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for row_num, row in enumerate(csv.DictReader(handle), start=2):
            item_id = row["item_id"]
            if item_id not in pool_by_id:
                errors.append(f"row {row_num}: unknown item_id {item_id}")
                continue
            scores = {}
            ok = True
            for dim in DIMENSIONS:
                raw = row[dim].strip()
                if not raw.isdigit() or not (1 <= int(raw) <= 5):
                    errors.append(f"row {row_num} ({item_id}): {dim}={raw!r} is not 1-5")
                    ok = False
                    continue
                scores[dim] = int(raw)
            if not ok:
                continue
            ratings.append(
                {
                    "item_id": item_id,
                    "system": pool_by_id[item_id]["system"],
                    "scores": scores,
                    "rating_source": "human",
                    "human_reviewed": True,
                }
            )

    if errors:
        print(f"{len(errors)} row(s) could not be imported — fix these in the CSV and re-run:")
        for error in errors[:30]:
            print(f"  - {error}")
        if not ratings:
            return
        print(f"\nImporting the {len(ratings)} valid row(s) anyway; fix and re-import to complete the rest.")

    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RATINGS_PATH.open("w", encoding="utf-8") as handle:
        for rating in ratings:
            handle.write(json.dumps(rating) + "\n")
    print(f"Imported {len(ratings)} human reply ratings -> {RATINGS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "import"])
    args = parser.parse_args()
    if args.action == "export":
        export_csv()
    else:
        import_csv()


if __name__ == "__main__":
    main()
