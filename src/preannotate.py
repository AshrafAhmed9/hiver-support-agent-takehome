"""Generates pre-annotator label suggestions for the golden candidate pool.

Uses PRE_ANNOTATOR_MODEL (Qwen family via Groq — a different family from both
the generator and the judge, see src/config.py) to suggest an intent for 150
of the 200 candidates. The other 50 are held blind (no suggestion field) so
the labelling session can measure anchoring bias: compare human-vs-suggestion
agreement on the suggested items against human-vs-suggestion agreement on the
blind items (computed post-hoc, blind items just never show a suggestion).

This step only writes suggestions to data/golden/golden_candidates.jsonl. It
never writes to golden_v1.jsonl or labeling_log.jsonl — those only get
written by a human running src/label_tui.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import INTENTS, PRE_ANNOTATOR_MODEL, RANDOM_SEED
from src.llm import CachedLLM, groq_call_fn

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/golden/golden_candidates.jsonl"

PROMPT_TEMPLATE = """You are labelling a customer support tweet sent to @spotifycares with exactly one intent from this fixed list:

{intents}

Definitions:
- playback_or_app_bug: app/playback misbehaves (crashes, skips, freezes, controls not working)
- login_or_account_access: can't log in, password reset, account locked/hacked
- billing_or_subscription_charge: wrong charge, failed promo, refund request, upgrade/downgrade
- content_availability_or_licensing: specific song/album missing, region-locked, or removed
- feature_request_or_product_feedback: wants a feature that doesn't exist, or complains about a design choice
- device_or_platform_compatibility: works on one device/OS but not another, device context is central
- account_data_or_privacy: wants account/data removed, objects to data use, language/geolocation settings ignored
- how_to_or_usage_question: asks how to do something the product already supports
- other: a genuine support request that fits none of the above
- not_a_support_request: praise, thanks, jokes, song requests, bare mentions, not an actual request

Message: "{text}"

Respond with only the intent label, nothing else."""


def suggest_intent(llm: CachedLLM, text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(intents="\n".join(f"- {i}" for i in INTENTS), text=text)
    response = llm.generate(prompt, {"temperature": 0.0})
    label = response.text.strip().lower().replace(" ", "_")
    return label if label in INTENTS else "other"


def annotate(blind_fraction: float = 0.25, seed: int = RANDOM_SEED, allow_live: bool = True) -> None:
    records = [json.loads(line) for line in CANDIDATES_PATH.open(encoding="utf-8") if line.strip()]
    rng = random.Random(seed)
    ids = [r["id"] for r in records]
    rng.shuffle(ids)
    n_blind = round(len(ids) * blind_fraction)
    blind_ids = set(ids[:n_blind])

    llm = CachedLLM(PRE_ANNOTATOR_MODEL, groq_call_fn(PRE_ANNOTATOR_MODEL), allow_live=allow_live)
    for record in records:
        record["blind"] = record["id"] in blind_ids
        if record["blind"]:
            record["suggested_intent"] = None
        else:
            record["suggested_intent"] = suggest_intent(llm, record["customer_text"])

    with CANDIDATES_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    n_suggested = sum(1 for r in records if not r["blind"])
    print(f"Annotated {n_suggested} / {len(records)} candidates. {n_blind} held blind.")


if __name__ == "__main__":
    annotate()
