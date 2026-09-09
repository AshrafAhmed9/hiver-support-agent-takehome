from src.agent import EvidenceCitation, GeneratedDraft, run_agent
from src.retrieve import BM25Retriever


def retriever():
    return BM25Retriever(
        [{"message_id": "a", "customer_text": "music skips", "historical_reply_text": "Try restarting the app."}]
    )


def test_agent_allows_cited_general_guidance():
    result = run_agent(
        "music skips",
        retriever(),
        lambda _text, _evidence: GeneratedDraft(
            intent="playback", reply_draft="Try restarting the app.", evidence=[EvidenceCitation(source_message_id="a", supporting_span="Try restarting the app.")]
        ),
    )
    assert result.route == "auto"


def test_agent_fails_closed_on_fabricated_evidence_citation():
    result = run_agent(
        "music skips",
        retriever(),
        lambda _text, _evidence: GeneratedDraft(
            intent="playback", reply_draft="Try restarting the app.", evidence=[EvidenceCitation(source_message_id="a", supporting_span="We refunded you.")]
        ),
    )
    assert result.route == "escalate"
    assert result.reason_codes == ("INVALID_EVIDENCE_CITATION",)


def test_agent_fails_closed_when_generator_errors():
    def fails(_text, _evidence):
        raise TimeoutError

    result = run_agent("music skips", retriever(), fails)
    assert result.reason_codes == ("GENERATION_FAILURE",)
