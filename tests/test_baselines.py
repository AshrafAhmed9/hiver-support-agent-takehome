from src.baselines import CANNED_REPLY, TfidfIntentClassifier, TrivialBaseline, majority_intent


def test_majority_intent():
    assert majority_intent(["a", "b", "a", "a", "c"]) == "a"


def test_trivial_baseline_fixed_route_and_canned_reply():
    baseline = TrivialBaseline(["billing", "billing", "login"], route="escalate")
    pred = baseline.predict("anything")
    assert pred.intent == "billing"
    assert pred.reply_draft == CANNED_REPLY
    assert pred.route == "escalate"


def test_trivial_baseline_rejects_invalid_route():
    import pytest

    with pytest.raises(ValueError):
        TrivialBaseline(["a"], route="auto_maybe")


def test_tfidf_classifier_fits_and_predicts_seen_pattern():
    texts = ["I can't log in to my account", "my password is wrong", "I was charged twice", "refund my subscription"]
    intents = ["login", "login", "billing", "billing"]
    clf = TfidfIntentClassifier().fit(texts, intents)
    preds = clf.predict(["I forgot my password"])
    assert preds[0] in {"login", "billing"}  # sanity: valid label space, not crashing
