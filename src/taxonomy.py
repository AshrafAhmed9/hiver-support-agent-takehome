"""Clustering aid used to discover candidate intents. Not the taxonomy itself.

Run this, read the printed cluster exemplars, then a human writes the intent
definitions and edge cases by hand into data/golden/codebook.md. Reporting
"I ran k-means and used the clusters as my labels" is a weak answer to the
obvious interview question, so this script is explicitly a discovery step.

Pure scikit-learn (TF-IDF + TruncatedSVD + KMeans) so it runs with no torch
dependency, matching the embedding-backend fallback described in the plan.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]


def load_texts(path: Path, sample_size: int, seed: int) -> list[str]:
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    texts = [r["customer_text"] for r in records if r.get("customer_text")]
    if len(texts) > sample_size:
        texts = random.Random(seed).sample(texts, k=sample_size)
    return texts


def cluster_and_print(texts: list[str], k_range: range, top_n: int = 12) -> None:
    vectorizer = TfidfVectorizer(max_features=5000, min_df=3, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(100, tfidf.shape[1] - 1), random_state=0)
    reduced = svd.fit_transform(tfidf)

    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(reduced)
        score = silhouette_score(reduced, labels, sample_size=min(2000, len(texts)), random_state=0)
        print(f"k={k:>2}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"\n=== best k={best_k} (silhouette={best_score:.4f}) ===\n")
    centers = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(reduced)
    distances = centers.transform(reduced)
    for cluster_id in range(best_k):
        member_idx = [i for i, lbl in enumerate(best_labels) if lbl == cluster_id]
        member_idx.sort(key=lambda i: distances[i][cluster_id])
        print(f"--- cluster {cluster_id} (n={len(member_idx)}) ---")
        for i in member_idx[:top_n]:
            print(" ", texts[i][:140].replace("\n", " "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/interim/SpotifyCares_train.jsonl")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    texts = load_texts(args.input, args.sample_size, args.seed)
    cluster_and_print(texts, range(args.k_min, args.k_max + 1))


if __name__ == "__main__":
    main()
