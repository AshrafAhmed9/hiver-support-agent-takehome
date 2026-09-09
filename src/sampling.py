"""Builds the golden-set candidate pool for human labelling.

This module never assigns final labels. It does two cheap, non-LLM things:

1. A TF-IDF nearest-centroid pseudo-labeller, fit only on the codebook's own
   worked examples (data/golden/codebook.md), used purely to *stratify* the
   sample across intents so rare intents aren't drowned out. This is the
   "first-pass cheap classifier" mentioned in the codebook's sampling note.
2. A stratified, seeded sample from the held-out test_pool split, with hard
   /ambiguous cases (low nearest-centroid margin) oversampled.

The pre-annotator LLM suggestion (shown to the human labeller for 150 of the
200 items, per src/label_tui.py) is a separate, later step — kept separate so
this module has no LLM dependency and is trivially testable.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import INTENTS, RANDOM_SEED

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_PATH = ROOT / "data/golden/codebook.md"
TEST_POOL_PATH = ROOT / "data/interim/SpotifyCares_test_pool.jsonl"
CANDIDATES_PATH = ROOT / "data/golden/golden_candidates.jsonl"

# Worked examples pulled from the codebook (kept in code, not re-parsed from
# markdown, so this module has no fragile markdown-scraping dependency).
CODEBOOK_EXAMPLES: dict[str, list[str]] = {
    "playback_or_app_bug": [
        "it will play about 2 songs and cut off wont let me press shuffle play again",
        "the lockscreen controls and out of app pause functionality have not been functional since the latest update",
        "app crashes every time I try to open a playlist",
        "songs keep skipping every 10 seconds",
    ],
    "login_or_account_access": [
        "why can't I login to my account, username and password are right",
        "my premium account is being hacked and used right now",
        "I reset my password but now I can't get back into my account",
        "an unknown device has connected to my spotify account",
    ],
    "billing_or_subscription_charge": [
        "I got charged 3 times for the student promo",
        "I paid for premium but it still shows free",
        "why was I charged after my trial should have ended",
        "I want a refund for the last 2 months I couldn't use the app",
    ],
    "content_availability_or_licensing": [
        "why did you delete this album",
        "why aren't these tracks available in my country",
        "this song shows unavailable when I try to play it",
        "the album disappeared from my library",
    ],
    "feature_request_or_product_feedback": [
        "why can't I block explicit songs",
        "when are you going to support chromecast",
        "please add a way to send songs to friends",
        "the new playlist format is a struggle to use, please change it back",
    ],
    "device_or_platform_compatibility": [
        "on android 7 the app disconnects but works fine restarting",
        "spotify crashes only on my ios device not on desktop",
        "the app doesn't work on my smart tv but works on my phone",
    ],
    "account_data_or_privacy": [
        "please remove my account, my email was used without permission",
        "please honor my language settings and stop using geolocation",
        "I want my data deleted from your systems",
    ],
    "how_to_or_usage_question": [
        "how do I change my profile picture",
        "how do I download a playlist for offline listening",
        "how do I switch from trial to paid premium",
    ],
    "other": [
        "why is it so hard to find a contact number for support",
        "can someone from your team actually help me",
    ],
    "not_a_support_request": [
        "thanks so much you're the best",
        "I need this song added right now",
        "just a random tweet mentioning the brand with no request",
    ],
}


def fit_pseudo_labeller() -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    labels: list[str] = []
    texts: list[str] = []
    for intent, examples in CODEBOOK_EXAMPLES.items():
        for example in examples:
            texts.append(example)
            labels.append(intent)
    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(texts).toarray()
    centroids = np.zeros((len(INTENTS), matrix.shape[1]))
    for intent in INTENTS:
        rows = [i for i, lbl in enumerate(labels) if lbl == intent]
        if rows:
            centroids[INTENTS.index(intent)] = matrix[rows].mean(axis=0)
    return vectorizer, centroids, INTENTS


def pseudo_label(vectorizer: TfidfVectorizer, centroids: np.ndarray, intents: list[str], text: str) -> tuple[str, float]:
    vec = vectorizer.transform([text]).toarray()[0]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return "other", 0.0
    sims = centroids @ vec / (np.linalg.norm(centroids, axis=1) * norm + 1e-9)
    order = np.argsort(-sims)
    top, second = order[0], order[1]
    margin = float(sims[top] - sims[second])
    return intents[top], margin


HARD_HINT_RE = re.compile(r"\?|\bbut\b|\bhowever\b|\band also\b", re.IGNORECASE)


def build_candidate_pool(
    pool_path: Path = TEST_POOL_PATH,
    per_intent: int = 20,
    hard_bonus: int = 6,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Stratified sample: `per_intent` per bucket, plus `hard_bonus` extra
    low-margin (ambiguous) items per bucket. Yields ~200 for 10 intents."""
    records = [json.loads(line) for line in pool_path.open(encoding="utf-8") if line.strip()]
    vectorizer, centroids, intents = fit_pseudo_labeller()

    by_intent: dict[str, list[tuple[dict, float]]] = {intent: [] for intent in intents}
    for record in records:
        text = record.get("customer_text", "")
        if not text:
            continue
        intent, margin = pseudo_label(vectorizer, centroids, intents, text)
        by_intent[intent].append((record, margin))

    rng = random.Random(seed)
    pool: list[dict] = []
    for intent in intents:
        items = by_intent[intent]
        if not items:
            continue
        rng.shuffle(items)
        # Easy/typical examples: higher margin first.
        items_by_margin = sorted(items, key=lambda pair: -pair[1])
        confident = items_by_margin[: max(0, per_intent - hard_bonus)]
        ambiguous = sorted(items, key=lambda pair: pair[1])[:hard_bonus]
        chosen = {record["message_id"]: (record, margin) for record, margin in confident + ambiguous}
        for record, margin in chosen.values():
            pool.append(
                {
                    "id": f"g_{record['message_id']}",
                    "message_id": record["message_id"],
                    "customer_text": record["customer_text"],
                    "context": record.get("context", []),
                    "created_at": record.get("created_at"),
                    "stratify_intent": intent,
                    "difficulty": "hard" if margin < 0.05 else "medium" if margin < 0.15 else "easy",
                }
            )
    rng.shuffle(pool)
    return pool


def write_candidate_pool() -> None:
    pool = build_candidate_pool()
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8") as handle:
        for item in pool:
            handle.write(json.dumps(item) + "\n")
    print(f"Wrote {len(pool)} candidates to {CANDIDATES_PATH}")


if __name__ == "__main__":
    write_candidate_pool()
