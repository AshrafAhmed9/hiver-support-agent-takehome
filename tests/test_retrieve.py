import json

from src.policy import assess_draft
from src.retrieve import BM25Retriever, load_historical_pairs


def test_bm25_returns_relevant_historical_example():
    retriever = BM25Retriever(
        [
            {"message_id": "a", "customer_text": "Spotify keeps skipping songs", "historical_reply_text": "Restart the app."},
            {"message_id": "b", "customer_text": "Where is my playlist?", "historical_reply_text": "Check your library."},
        ]
    )
    assert retriever.search("songs are skipping in Spotify", 1)[0].message_id == "a"


def test_corpus_sampling_is_deterministic_and_excludes_missing_replies(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps({"message_id": str(i), "customer_text": "x", "historical_reply_text": "r" if i else None}) for i in range(10)))
    first = load_historical_pairs(path, limit=3, seed=4)
    second = load_historical_pairs(path, limit=3, seed=4)
    assert first == second
    assert len(first) == 3
    assert all(item["historical_reply_text"] for item in first)


def test_policy_escalates_account_commitments_despite_evidence():
    decision = assess_draft("We'll refund your subscription today.", has_evidence=True, predicted_intent="billing")
    assert decision.route == "escalate"
    assert "ACCOUNT_ACTION_OR_COMMITMENT" in decision.reason_codes


def test_policy_allows_general_evidence_grounded_guidance():
    assert assess_draft("Try restarting the app, then log in again.", has_evidence=True, predicted_intent="playback").route == "auto"
