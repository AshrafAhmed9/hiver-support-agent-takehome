"""Small, local retrieval over frozen historical support examples."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Evidence:
    message_id: str
    customer_text: str
    brand_reply: str
    score: float


def load_historical_pairs(path: Path, limit: int = 5_000, seed: int = 20260909) -> list[dict]:
    """Take a frozen, deterministic corpus sample without using final labels."""
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    eligible = [item for item in records if item.get("historical_reply_text")]
    if len(eligible) <= limit:
        return eligible
    # Sort before sampling so input filesystem order cannot change the corpus.
    eligible.sort(key=lambda item: item["message_id"])
    return random.Random(seed).sample(eligible, k=limit)


class BM25Retriever:
    def __init__(self, records: list[dict]) -> None:
        if not records:
            raise ValueError("retrieval corpus cannot be empty")
        self.records = records
        self.index = BM25Okapi([tokenize(item["customer_text"]) for item in records])
        corpus_ids = "\n".join(sorted(item["message_id"] for item in records))
        self.corpus_hash = hashlib.sha256(corpus_ids.encode()).hexdigest()

    def search(self, query: str, limit: int = 5) -> list[Evidence]:
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], self.records[item[0]]["message_id"]))
        return [
            Evidence(
                message_id=self.records[index]["message_id"],
                customer_text=self.records[index]["customer_text"],
                brand_reply=self.records[index]["historical_reply_text"],
                score=float(score),
            )
            for index, score in ranked[:limit]
        ]

