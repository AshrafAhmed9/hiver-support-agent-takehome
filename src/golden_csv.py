"""CSV export/import for labelling the golden set.

Export writes one blank row per candidate — id, difficulty, customer text
and prior context, with the three label columns empty. Labels are written
by hand against data/golden/codebook.md. Import validates every row and
writes data/golden/golden_v1.jsonl.

Workflow:
    uv run python -m src.golden_csv export   # writes data/golden/golden_labelling.csv
    ... fill in final_intent / final_should_escalate / final_escalate_reason ...
    uv run python -m src.golden_csv import   # writes golden_v1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import INTENTS

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/golden/golden_candidates.jsonl"
CSV_PATH = ROOT / "data/golden/golden_labelling.csv"
GOLDEN_PATH = ROOT / "data/golden/golden_v1.jsonl"
LOG_PATH = ROOT / "data/golden/labeling_log.jsonl"

FIELDNAMES = [
    "id",
    "difficulty",
    "customer_text",
    "prior_context",
    "final_intent",
    "final_should_escalate",
    "final_escalate_reason",
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def export_csv() -> None:
    candidates = _load_jsonl(CANDIDATES_PATH)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in candidates:
            context = " | ".join(turn.get("text", "") for turn in record.get("context") or [])
            writer.writerow(
                {
                    "id": record["id"],
                    "difficulty": record.get("difficulty", ""),
                    "customer_text": record["customer_text"],
                    "prior_context": context,
                    "final_intent": "",
                    "final_should_escalate": "",
                    "final_escalate_reason": "",
                }
            )
    print(f"Wrote {len(candidates)} rows to {CSV_PATH}.")
    print("Fill in final_intent / final_should_escalate / final_escalate_reason for every row,")
    print("then run: uv run python -m src.golden_csv import")


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from {raw!r}")


def import_csv() -> None:
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} does not exist. Run `export` first.")
        return
    candidates_by_id = {r["id"]: r for r in _load_jsonl(CANDIDATES_PATH)}

    golden_records = []
    log_records = []
    errors = []
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for row_num, row in enumerate(csv.DictReader(handle), start=2):
            item_id = row["id"]
            source = candidates_by_id.get(item_id)
            if source is None:
                errors.append(f"row {row_num}: unknown id {item_id}")
                continue

            final_intent = row["final_intent"].strip()
            if final_intent not in INTENTS:
                errors.append(f"row {row_num} ({item_id}): final_intent {final_intent!r} is not a valid intent")
                continue
            try:
                final_should_escalate = _parse_bool(row["final_should_escalate"])
            except ValueError as exc:
                errors.append(f"row {row_num} ({item_id}): {exc}")
                continue
            final_reason = row["final_escalate_reason"].strip()
            if not final_reason:
                errors.append(f"row {row_num} ({item_id}): final_escalate_reason is empty")
                continue

            log_records.append(
                {
                    "id": item_id,
                    "final_intent": final_intent,
                    "final_should_escalate": final_should_escalate,
                    "final_escalate_reason": final_reason,
                }
            )
            golden_records.append(
                {
                    "id": item_id,
                    "thread_id": source["message_id"],
                    "customer_text": source["customer_text"],
                    "context": source.get("context", []),
                    "intent": final_intent,
                    "secondary_intent": None,
                    "should_escalate": final_should_escalate,
                    "escalate_reason": final_reason,
                    "difficulty": source.get("difficulty", "medium"),
                    "label_source": "human",
                }
            )

    if errors:
        print(f"{len(errors)} row(s) could not be imported — fix these in the CSV and re-run:")
        for error in errors[:30]:
            print(f"  - {error}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        if not golden_records:
            return
        print(f"\nImporting the {len(golden_records)} valid row(s) anyway; fix and re-import to complete the rest.")

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as handle:
        for record in golden_records:
            handle.write(json.dumps(record) + "\n")
    with LOG_PATH.open("w", encoding="utf-8") as handle:
        for record in log_records:
            handle.write(json.dumps(record) + "\n")

    print(f"Imported {len(golden_records)} labelled items -> {GOLDEN_PATH}")


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
