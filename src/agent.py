"""Typed orchestration for the proposed retrieval-grounded support agent.

Provider invocation is deliberately separate from this module so tests can use a
deterministic callable and final evaluations can record exact request metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import BRAND, GENERATOR_MODEL, INTENTS
from src.llm import CachedLLM, groq_call_fn
from src.policy import assess_draft
from src.retrieve import BM25Retriever, Evidence, load_historical_pairs


class EvidenceCitation(BaseModel):
    source_message_id: str
    supporting_span: str = Field(min_length=1, max_length=500)


class GeneratedDraft(BaseModel):
    intent: str
    reply_draft: str = Field(min_length=1, max_length=800)
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=5)

    @field_validator("intent")
    @classmethod
    def valid_intent(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("intent must be non-empty and trimmed")
        return value


@dataclass(frozen=True)
class AgentResult:
    intent: str
    reply_draft: str
    evidence: tuple[EvidenceCitation, ...]
    route: str
    reason_codes: tuple[str, ...]


Generator = Callable[[str, list[Evidence]], GeneratedDraft]


def validate_citations(draft: GeneratedDraft, retrieved: list[Evidence]) -> bool:
    """Citation IDs and quoted spans must be present in the retrieved evidence."""
    evidence_by_id = {item.message_id: item for item in retrieved}
    for citation in draft.evidence:
        source = evidence_by_id.get(citation.source_message_id)
        if source is None or citation.supporting_span not in source.brand_reply:
            return False
    return True


def run_agent(customer_text: str, retriever: BM25Retriever, generate: Generator) -> AgentResult:
    retrieved = retriever.search(customer_text, limit=5)
    try:
        generated = generate(customer_text, retrieved)
    except (ConnectionError, TimeoutError, ValueError):
        return AgentResult(
            intent="other",
            reply_draft="Thanks for getting in touch. A support specialist will review this.",
            evidence=(),
            route="escalate",
            reason_codes=("GENERATION_FAILURE",),
        )
    citations_valid = validate_citations(generated, retrieved)
    decision = assess_draft(
        generated.reply_draft,
        has_evidence=bool(retrieved) and citations_valid,
        predicted_intent=generated.intent,
    )
    reasons = decision.reason_codes if citations_valid else ("INVALID_EVIDENCE_CITATION",)
    return AgentResult(
        intent=generated.intent,
        reply_draft=generated.reply_draft,
        evidence=tuple(generated.evidence),
        route="escalate" if not citations_valid else decision.route,
        reason_codes=reasons,
    )


GENERATION_PROMPT = """You are drafting a public reply for the @{brand} support account on Twitter.
Only propose remedies that are directly supported by the evidence below — never invent a remedy,
a policy, or a commitment (refund, compensation, specific date) that isn't in the evidence.

Fixed intent list (choose exactly one): {intents}

Customer message: "{customer_text}"

Evidence (historical resolved threads for this brand):
{evidence}

Respond with ONLY a JSON object of this exact shape:
{{"intent": "<one of the fixed intents>", "reply_draft": "<your reply, max 800 chars>",
  "evidence": [{{"source_message_id": "<id from evidence above>", "supporting_span": "<exact substring from that evidence's reply>"}}]}}
Cite 1-3 evidence items. If nothing in the evidence is relevant, still return valid JSON with
intent "other" or "not_a_support_request" and an empty-effort reply_draft, but you MUST cite at
least one evidence item's id with any short substring of its text (routing will treat this as
low-confidence and escalate)."""


def make_groq_generator(model_id: str = GENERATOR_MODEL, allow_live: bool = True) -> Generator:
    """Builds a Generator callable backed by Groq, going through the disk
    cache in src/llm.py so repeated runs are free and offline-replayable."""
    llm = CachedLLM(model_id, groq_call_fn(model_id), allow_live=allow_live)

    def generate(customer_text: str, retrieved: list[Evidence]) -> GeneratedDraft:
        evidence_block = "\n".join(f"- id={e.message_id}: {e.brand_reply}" for e in retrieved) or "(none retrieved)"
        prompt = GENERATION_PROMPT.format(
            brand=BRAND, intents=", ".join(INTENTS), customer_text=customer_text, evidence=evidence_block
        )
        response = llm.generate(prompt, {"temperature": 0.0})
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not match:
            raise ValueError(f"generator returned no JSON object: {response.text[:200]}")
        return GeneratedDraft.model_validate(json.loads(match.group()))

    return generate


def _cli_generate(customer_text: str, retrieved: list[Evidence]) -> GeneratedDraft:
    try:
        return make_groq_generator()(customer_text, retrieved)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"generation/parsing failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: classify, draft, and route one customer message.")
    parser.add_argument("customer_text", help="the inbound customer message")
    parser.add_argument(
        "--corpus", type=Path, default=Path(__file__).resolve().parents[1] / f"data/interim/{BRAND}_train.jsonl"
    )
    args = parser.parse_args()

    corpus = load_historical_pairs(args.corpus)
    retriever = BM25Retriever(corpus)
    result = run_agent(args.customer_text, retriever, _cli_generate)

    print(f"intent:        {result.intent}")
    print(f"route:         {result.route}")
    print(f"reason_codes:  {', '.join(result.reason_codes) or '(none)'}")
    print("evidence:")
    for citation in result.evidence:
        print(f"  - {citation.source_message_id}: \"{citation.supporting_span}\"")
    print("reply_draft:")
    print(f"  {result.reply_draft}")


if __name__ == "__main__":
    main()
