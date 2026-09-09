"""Deterministic capability rules for public drafting.

These rules are intentionally conservative: a historical example of a refund or
account action never grants authority to promise it to a new customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ACCOUNT_ACTION = re.compile(
    r"\b(refund|refund(?:ed|ing)?|compensation|reimburse|credit your|chargeback|"
    r"cancel(?:led|lation)?|delete (?:your )?account|verify (?:your )?identity)\b",
    re.IGNORECASE,
)
SENSITIVE_DATA = re.compile(r"\b(password|card number|cvv|social security|confirmation number)\b", re.IGNORECASE)
PRIVATE_HANDOFF = re.compile(r"\b(?:dm|direct message|private message|send us a message)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyDecision:
    route: str
    reason_codes: tuple[str, ...]


def assess_draft(draft: str, *, has_evidence: bool, predicted_intent: str) -> PolicyDecision:
    reasons: list[str] = []
    if not has_evidence:
        reasons.append("NO_APPLICABLE_EVIDENCE")
    if predicted_intent in {"other_or_unclear", "not_support"}:
        reasons.append("INTENT_NOT_ELIGIBLE_FOR_AUTO")
    if ACCOUNT_ACTION.search(draft):
        reasons.append("ACCOUNT_ACTION_OR_COMMITMENT")
    if SENSITIVE_DATA.search(draft):
        reasons.append("SENSITIVE_DATA_REQUEST")
    if PRIVATE_HANDOFF.search(draft):
        reasons.append("PRIVATE_HANDOFF_REQUIRED")
    return PolicyDecision("escalate" if reasons else "auto", tuple(reasons))

