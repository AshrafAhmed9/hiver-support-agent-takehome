from src.config import INTENTS
from src.sampling import fit_pseudo_labeller, pseudo_label


def test_pseudo_label_returns_valid_intent_and_margin():
    vectorizer, centroids, intents = fit_pseudo_labeller()
    intent, margin = pseudo_label(vectorizer, centroids, intents, "why was I charged twice for premium")
    assert intent in INTENTS
    assert isinstance(margin, float)


def test_pseudo_label_handles_empty_text():
    vectorizer, centroids, intents = fit_pseudo_labeller()
    intent, margin = pseudo_label(vectorizer, centroids, intents, "")
    assert intent in INTENTS
    assert margin == 0.0
