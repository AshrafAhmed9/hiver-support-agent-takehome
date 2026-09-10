"""Trivial and simple baselines, required by the brief as comparison points.

Trivial: majority-class intent, one canned reply, and the two routing
extremes (escalate-everything / auto-everything) that bracket the cost model.

Simple, no LLM anywhere: TF-IDF + logistic regression for intent (trained on
train-split records only, never on golden), and a BM25-copy baseline that
returns the single closest historical brand reply verbatim. BM25-copy is
expected to beat the LLM on groundedness — it is real text a real agent sent
— which is a more informative comparison than a strawman.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.retrieve import BM25Retriever

CANNED_REPLY = (
    "Thanks for reaching out! A support specialist will take a look and follow up with you shortly."
)


@dataclass(frozen=True)
class BaselinePrediction:
    intent: str
    reply_draft: str
    route: str


def majority_intent(train_intents: list[str]) -> str:
    return Counter(train_intents).most_common(1)[0][0]


class TrivialBaseline:
    """Majority-class intent, one canned reply, fixed routing policy."""

    def __init__(self, train_intents: list[str], route: str) -> None:
        if route not in {"auto", "escalate"}:
            raise ValueError("route must be 'auto' or 'escalate'")
        self.intent = majority_intent(train_intents)
        self.route = route

    def predict(self, _customer_text: str) -> BaselinePrediction:
        return BaselinePrediction(intent=self.intent, reply_draft=CANNED_REPLY, route=self.route)


class TfidfIntentClassifier:
    """TF-IDF + logistic regression. No LLM. Fit only on labelled train data."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts: list[str], intents: list[str]) -> TfidfIntentClassifier:
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, intents)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.model.predict(self.vectorizer.transform(texts)))

    def predict_proba_entropy(self, texts: list[str]) -> list[float]:
        """Shannon entropy of the predicted class distribution, used as a
        confidence signal for the escalation combiner (§6 of the plan)."""
        import numpy as np

        probs = self.model.predict_proba(self.vectorizer.transform(texts))
        probs = np.clip(probs, 1e-12, 1.0)
        return list(-(probs * np.log(probs)).sum(axis=1))

    def confidence_threshold_route(self, texts: list[str], threshold: float) -> list[BaselinePrediction]:
        preds = self.predict(texts)
        proba = self.model.predict_proba(self.vectorizer.transform(texts))
        max_conf = proba.max(axis=1)
        return [
            BaselinePrediction(
                intent=intent,
                reply_draft=CANNED_REPLY,
                route="auto" if conf >= threshold else "escalate",
            )
            for intent, conf in zip(preds, max_conf)
        ]


class BM25CopyBaseline:
    """Copies the single closest historical brand reply verbatim. No generation."""

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    def predict(self, customer_text: str) -> BaselinePrediction:
        hits = self.retriever.search(customer_text, limit=1)
        reply = hits[0].brand_reply if hits else CANNED_REPLY
        # No confidence signal available; route auto only when a hit exists.
        return BaselinePrediction(intent="unknown", reply_draft=reply, route="auto" if hits else "escalate")


def load_intent_labels(path: Path) -> list[dict]:
    """Loads records with an `intent` field.

    Used to fit the TF-IDF classifier on data disjoint from the golden set.
    """
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
