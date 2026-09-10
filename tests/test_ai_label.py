import json

import pytest

from src.ai_label import (
    build_prompt,
    enforce_fixed_policy,
    parse_batch,
    validated_response,
    visible_input,
)
from src.llm import LLMResponse


def record():
    return {"id": "g_1", "customer_text": "Music skips", "context": [],
            "suggested_intent": "SECRET_SUGGESTION", "historical_reply_text": "SECRET_FUTURE"}


def label():
    return {"id": "g_1", "intent": "playback_or_app_bug", "secondary_intent": None,
            "should_escalate": False, "escalate_reason": "Basic public troubleshooting is possible.",
            "uncertain": False, "uncertainty_reason": ""}


def test_prompt_excludes_old_suggestions_and_future_replies():
    prompt = build_prompt([record()], "Taxonomy")
    assert "SECRET_SUGGESTION" not in prompt
    assert "SECRET_FUTURE" not in prompt
    assert set(visible_input(record())) == {"id", "customer_text", "context"}


def test_batch_rejects_missing_or_repeated_ids():
    for labels in ([], [label(), label()], [{**label(), "id": "wrong"}]):
        with pytest.raises(ValueError):
            parse_batch(json.dumps({"labels": labels}), [record()])


def test_batch_rejects_invalid_labels_and_nonboolean_routes():
    for change in ({"intent": "invented"}, {"should_escalate": "false"},
                   {"uncertain": True}, {"secondary_intent": "playback_or_app_bug"}):
        with pytest.raises(ValueError):
            parse_batch(json.dumps({"labels": [{**label(), **change}]}), [record()])


def test_batch_accepts_complete_valid_output():
    assert parse_batch(json.dumps({"labels": [label()]}), [record()]) == [label()]
    assert parse_batch(json.dumps([label()]), [record()]) == [label()]


def test_policy_normalization_preserves_original_model_label_and_is_idempotent():
    raw = {**label(), "intent": "not_a_support_request"}
    result = enforce_fixed_policy(raw)
    assert result["should_escalate"] is True
    assert result["raw_model_label"] == raw
    assert result["raw_model_label"]["should_escalate"] is False
    assert result["routing_label_source"] == "deterministic_policy"
    assert result["policy_adjustments"] == ["MANDATORY_REVIEW_INTENT"]
    assert enforce_fixed_policy(result) == result


def test_general_guidance_routing_is_not_changed_by_fixed_policy():
    result = enforce_fixed_policy(label())
    assert result["should_escalate"] is False
    assert result["routing_label_source"] == "ai"
    assert not result["policy_adjustments"]


def test_validation_retry_requires_model_to_supply_missing_reason():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, params):
            self.calls += 1
            result = {**label(), "escalate_reason": ""} if self.calls == 1 else label()
            return LLMResponse(json.dumps({"labels": [result]}), "test", False)

    fake = FakeLLM()
    labels, prompt = validated_response(fake, "original", [record()])
    assert labels == [label()]
    assert fake.calls == 2
    assert "INCLUDING should_escalate=false" in prompt


def test_eval_loader_rejects_unfinished_ai_run(tmp_path, monkeypatch):
    from src.eval import run_eval

    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(run_eval, "GOLDEN_PATH", tmp_path / "human.jsonl")
    ai_path = tmp_path / "data/labels/golden_ai_v1.jsonl"
    ai_path.parent.mkdir(parents=True)
    ai_path.write_text(json.dumps({**label(), "label_source": "ai", "human_reviewed": False}) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        run_eval.load_golden()


def test_ai_ratings_are_never_reported_as_human_agreement():
    from src.eval.judge_agreement import DIMENSIONS, agreement_report

    ratings = [{"item_id": str(i), "rating_source": "ai", "scores": dict.fromkeys(DIMENSIONS, i)}
               for i in (1, 3, 5)]
    report = agreement_report(ratings, ratings, "test")
    assert report["reference_source"] == "ai"
    assert report["agreement_type"] == "AI–AI agreement"
    unknown = [{"item_id": r["item_id"], "scores": r["scores"]} for r in ratings]
    assert agreement_report(unknown, ratings, "test")["agreement_type"] == "unverified-reference agreement"
