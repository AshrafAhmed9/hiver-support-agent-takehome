"""CSV export/import for bulk review of the golden set.

Export writes one row per candidate with the pre-annotator's draft
(intent / should_escalate / reason) prefilled into `final_*` columns for
review. Editing = overwriting the ones you disagree with. Import validates
every row and writes data/golden/golden_v1.jsonl.

Every finalized record is tagged with an honest provenance field describing
how it was produced — this is not "written from scratch by a human," it's
"drafted by a model, reviewed and confirmed/corrected by a human." See
REPORT.md for why that distinction matters and is disclosed rather than
hidden.

Workflow:
    uv run python -m src.golden_csv export   # writes data/golden/golden_labelling.csv
    ... edit final_intent / final_should_escalate / final_escalate_reason ...
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
    "suggested_intent",
    "suggested_should_escalate",
    "suggested_escalate_reason",
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
            intent = record.get("suggested_intent", "")
            should_escalate = record.get("suggested_should_escalate", "")
            reason = record.get("suggested_escalate_reason", "")
            writer.writerow(
                {
                    "id": record["id"],
                    "difficulty": record.get("difficulty", ""),
                    "customer_text": record["customer_text"],
                    "prior_context": context,
                    "suggested_intent": intent,
                    "suggested_should_escalate": should_escalate,
                    "suggested_escalate_reason": reason,
                    # Prefilled with the draft so editing = overwriting disagreements.
                    "final_intent": intent,
                    "final_should_escalate": should_escalate,
                    "final_escalate_reason": reason,
                }
            )
    print(f"Wrote {len(candidates)} rows to {CSV_PATH}.")
    print("Review/edit final_intent / final_should_escalate / final_escalate_reason for every row,")
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

            suggested_intent = row["suggested_intent"].strip() or None
            suggested_escalate_raw = row["suggested_should_escalate"].strip()
            suggested_escalate = _parse_bool(suggested_escalate_raw) if suggested_escalate_raw else None
            suggested_reason = row["suggested_escalate_reason"].strip() or None

            intent_overridden = suggested_intent is not None and final_intent != suggested_intent
            escalate_overridden = suggested_escalate is not None and final_should_escalate != suggested_escalate
            reason_overridden = suggested_reason is not None and final_reason != suggested_reason

            log_records.append(
                {
                    "id": item_id,
                    "suggested_intent": suggested_intent,
                    "final_intent": final_intent,
                    "intent_overridden": intent_overridden,
                    "suggested_should_escalate": suggested_escalate,
                    "final_should_escalate": final_should_escalate,
                    "escalate_overridden": escalate_overridden,
                    "suggested_escalate_reason": suggested_reason,
                    "final_escalate_reason": final_reason,
                    "reason_overridden": reason_overridden,
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
                    # Honest provenance: drafted by a model, reviewed/confirmed or
                    # corrected by a human. Not "written from scratch by a human."
                    "label_source": "ai_drafted_human_reviewed",
                    "human_reviewed": True,
                    "intent_overridden": intent_overridden,
                    "escalate_overridden": escalate_overridden,
                    "reason_overridden": reason_overridden,
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

    n_overridden_intent = sum(1 for r in log_records if r["intent_overridden"])
    n_overridden_escalate = sum(1 for r in log_records if r["escalate_overridden"])
    n_overridden_reason = sum(1 for r in log_records if r["reason_overridden"])
    print(f"Imported {len(golden_records)} labelled items -> {GOLDEN_PATH}")
    print(
        f"Override rates: intent {n_overridden_intent}/{len(log_records)}, "
        f"escalate {n_overridden_escalate}/{len(log_records)}, "
        f"reason {n_overridden_reason}/{len(log_records)}"
    )


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
