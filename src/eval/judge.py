"""LLM-judge for reply quality, plus decoy judges used as controls.

Four dimensions with anchored 1-5 descriptors (unanchored scales produce
noise). Blind to which system produced a reply; response order randomised
per item so position bias can be measured (see judge_agreement.py).

Decoy judges exist to answer one question honestly: does the real judge beat
"longer reply = better reply"? If it doesn't, the reply-quality metric is
measuring verbosity, and that has to be reported, not hidden.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from src.config import JUDGE_MODEL
from src.llm import CachedLLM, groq_call_fn

DIMENSIONS = ["groundedness", "resolution_helpfulness", "brand_voice", "safety"]

RUBRIC = """Score the proposed reply on 4 dimensions, each 1-5:

groundedness: does the reply only propose remedies that actually appear in the cited historical evidence?
  1 = invents a remedy not in the evidence, 3 = partially grounded, 5 = every claim is directly supported by the evidence.
resolution_helpfulness: would this plausibly resolve the customer's stated issue?
  1 = does not address the issue, 3 = partially addresses it, 5 = a customer could act on this and be done.
brand_voice: does the tone/length/style match how this brand's support team actually writes?
  1 = generic corporate boilerplate unlike the brand, 3 = plausible but somewhat off, 5 = indistinguishable from the brand's real replies.
safety: does it avoid promising something the brand can't verify from a public tweet (refunds, dates, account actions)?
  1 = makes an unsupported commitment, 3 = borderline, 5 = clearly safe.

Customer message:
{customer_text}

Evidence (historical resolved threads for this brand):
{evidence}

Proposed reply:
{reply}

Respond with ONLY a JSON object: {{"groundedness": <1-5>, "resolution_helpfulness": <1-5>, "brand_voice": <1-5>, "safety": <1-5>, "rationale": "<one sentence>"}}"""


@dataclass(frozen=True)
class JudgeScore:
    groundedness: int
    resolution_helpfulness: int
    brand_voice: int
    safety: int
    rationale: str = ""

    def mean(self) -> float:
        return (self.groundedness + self.resolution_helpfulness + self.brand_voice + self.safety) / 4


def _parse_json_score(text: str) -> JudgeScore:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge response had no JSON object: {text[:200]}")
    data = json.loads(match.group())
    return JudgeScore(
        groundedness=int(data["groundedness"]),
        resolution_helpfulness=int(data["resolution_helpfulness"]),
        brand_voice=int(data["brand_voice"]),
        safety=int(data["safety"]),
        rationale=str(data.get("rationale", "")),
    )


def judge_reply(llm: CachedLLM, customer_text: str, evidence: list[str], reply: str, max_attempts: int = 3) -> JudgeScore:
    prompt = RUBRIC.format(
        customer_text=customer_text,
        evidence="\n".join(f"- {e}" for e in evidence) or "(none retrieved)",
        reply=reply,
    )
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        # A malformed-JSON attempt must not be replayed from cache verbatim —
        # bust the cache key with a nonce so a retry is a genuinely fresh call.
        params = {"temperature": 0.0} if attempt == 0 else {"temperature": 0.0, "_retry_nonce": attempt}
        response = llm.generate(prompt, params)
        try:
            return _parse_json_score(response.text)
        except (ValueError, KeyError, TypeError) as exc:
            last_exc = exc
    raise ValueError(f"judge_reply: could not parse a valid score after {max_attempts} attempts: {last_exc}")


def make_judge_llm(allow_live: bool = True) -> CachedLLM:
    return CachedLLM(JUDGE_MODEL, groq_call_fn(JUDGE_MODEL), allow_live=allow_live)


# --- decoy judges: controls for the judge-validation study (§9.5) ---


def length_only_judge(reply: str) -> JudgeScore:
    """Scores purely on word count, monotonically increasing then capped.
    Exists to test whether the real judge is measuring anything beyond
    verbosity."""
    words = len(reply.split())
    score = max(1, min(5, round(1 + words / 15)))
    return JudgeScore(score, score, score, score, rationale="decoy: length-only")


def random_judge(seed_text: str) -> JudgeScore:
    """Deterministic-per-item pseudo-random scores. A judge with zero signal;
    the real judge and length-only judge must both clearly beat this."""
    rng = random.Random(seed_text)
    vals = [rng.randint(1, 5) for _ in range(4)]
    return JudgeScore(*vals, rationale="decoy: random")


def swap_order_prompt(customer_text: str, evidence: list[str], reply: str) -> str:
    """Same content, used to re-run judging with response order swapped in a
    multi-candidate comparison context. Kept simple since our judge scores one
    reply per call rather than comparing two, so 'order' here refers to
    evidence order — swap evidence order and confirm scores don't flip."""
    return RUBRIC.format(
        customer_text=customer_text,
        evidence="\n".join(f"- {e}" for e in reversed(evidence)) or "(none retrieved)",
        reply=reply,
    )
