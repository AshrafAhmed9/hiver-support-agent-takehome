"""Generates draft label suggestions for the golden candidate pool.

Uses PRE_ANNOTATOR_MODEL (Qwen family via Groq — a different family from both
the generator and the judge, see src/config.py) to draft an intent, an
escalation call, and a one-line reason for every candidate. A human then
reviews and confirms or overrides every one — see data/golden/README or
REPORT.md for how that review was actually done.

This step only writes suggestions to data/golden/golden_candidates.jsonl. It
never writes to golden_v1.jsonl — that only happens after human review.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import INTENTS, PRE_ANNOTATOR_MODEL
from src.llm import CachedLLM, groq_call_fn

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/golden/golden_candidates.jsonl"

INTENT_PROMPT_TEMPLATE = """You are labelling a customer support tweet sent to @spotifycares with exactly one intent from this fixed list:

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

ESCALATE_PROMPT_TEMPLATE = """You are deciding whether a proposed public reply to this @spotifycares
customer support tweet could be safely auto-sent with NO human review and NO account access,
or whether it needs a human to handle it.

Escalate to a human (answer "escalate") if the message:
- requires account-specific access, verification, or a refund/compensation/billing action
- reports something that could be account security related (hacking, unauthorized access)
- is ambiguous, sarcastic, or not really a clear support request
- involves a promise the brand can't verify from a public tweet alone

It's fine to auto-handle (answer "auto") if the message is a general how-to question or a
known, generic troubleshooting issue (e.g. restart app, check app version) with no account
action needed.

Message: "{text}"

Respond with only one word: "escalate" or "auto"."""

REASON_PROMPT_TEMPLATE = """A customer support routing decision for this @spotifycares tweet was: {decision}.

Message: "{text}"

Write ONE short sentence (under 15 words) giving the concrete reason for that decision,
in plain language a support-ops reviewer would write. No preamble, just the sentence."""


def suggest_intent(llm: CachedLLM, text: str) -> str:
    prompt = INTENT_PROMPT_TEMPLATE.format(intents="\n".join(f"- {i}" for i in INTENTS), text=text)
    response = llm.generate(prompt, {"temperature": 0.0})
    label = response.text.strip().lower().replace(" ", "_")
    return label if label in INTENTS else "other"


def suggest_escalate(llm: CachedLLM, text: str) -> bool:
    prompt = ESCALATE_PROMPT_TEMPLATE.format(text=text)
    response = llm.generate(prompt, {"temperature": 0.0})
    return "escalate" in response.text.strip().lower()


def suggest_reason(llm: CachedLLM, text: str, should_escalate: bool) -> str:
    decision = "escalate to a human" if should_escalate else "auto-handle with no human review"
    prompt = REASON_PROMPT_TEMPLATE.format(decision=decision, text=text)
    response = llm.generate(prompt, {"temperature": 0.0})
    return response.text.strip().strip('"')


def annotate(allow_live: bool = True) -> None:
    records = [json.loads(line) for line in CANDIDATES_PATH.open(encoding="utf-8") if line.strip()]
    llm = CachedLLM(PRE_ANNOTATOR_MODEL, groq_call_fn(PRE_ANNOTATOR_MODEL), allow_live=allow_live)

    n_done = 0
    for record in records:
        if record.get("suggested_intent") is not None:
            continue  # already annotated (e.g. from the earlier 200-item run)
        record["suggested_intent"] = suggest_intent(llm, record["customer_text"])
        record["suggested_should_escalate"] = suggest_escalate(llm, record["customer_text"])
        record["suggested_escalate_reason"] = suggest_reason(
            llm, record["customer_text"], record["suggested_should_escalate"]
        )
        n_done += 1

    with CANDIDATES_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Drafted {n_done} new suggestion(s); {len(records)} candidates total have suggestions.")


if __name__ == "__main__":
    annotate()
