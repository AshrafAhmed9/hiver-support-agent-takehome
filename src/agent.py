"""Typed orchestration for the proposed retrieval-grounded support agent.

Provider invocation is deliberately separate from this module so tests can use a
deterministic callable and final evaluations can record exact request metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from src.policy import assess_draft
from src.retrieve import BM25Retriever, Evidence


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
            intent="other_or_unclear",
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
