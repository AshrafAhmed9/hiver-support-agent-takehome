"""Dataset ingestion, thread reconstruction, redaction, and chronological splits.

The functions in this module deliberately use only the standard library so they can
be tested before the model environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data/raw/twcs/twcs/twcs.csv"
INTERIM = ROOT / "data/interim"

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]?){7,}\d(?!\w)")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,15}")
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
LONG_ID_RE = re.compile(r"(?<!\w)\d{7,}(?!\w)")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Tweet:
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    response_ids: tuple[str, ...]
    in_response_to: str | None


@dataclass(frozen=True)
class Episode:
    message_id: str
    component_id: str
    created_at: str
    brand: str
    customer_text: str
    context: tuple[dict[str, str], ...]
    historical_reply_id: str | None
    historical_reply_text: str | None


def parse_id_list(raw: str | None) -> tuple[str, ...]:
    if not raw or raw.strip() in {"", "nan", "None"}:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_bool(raw: str | None) -> bool:
    return str(raw).strip().lower() == "true"


def parse_tweet(row: dict[str, str]) -> Tweet:
    raw_created = row["created_at"]
    try:
        created = datetime.fromisoformat(raw_created)
    except ValueError:
        created = datetime.strptime(raw_created, "%a %b %d %H:%M:%S %z %Y")
    return Tweet(
        tweet_id=str(row["tweet_id"]),
        author_id=row["author_id"],
        inbound=parse_bool(row.get("inbound")),
        created_at=created,
        text=row.get("text") or "",
        response_ids=parse_id_list(row.get("response_tweet_id")),
        in_response_to=(row.get("in_response_to_tweet_id") or None),
    )


def scrub_text(text: str, brand_handle: str) -> str:
    """Remove direct identifiers without erasing useful product/version numbers."""
    protected = brand_handle.lower().lstrip("@")

    def replace_handle(match: re.Match[str]) -> str:
        return f"@{protected}" if match.group()[1:].lower() == protected else "<USER>"

    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = LONG_ID_RE.sub("<ID>", text)
    text = HANDLE_RE.sub(replace_handle, text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_for_duplicate(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub("<url>", text)
    text = HANDLE_RE.sub("<user>", text)
    text = re.sub(r"[^a-z0-9<> ]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def shingles(text: str, width: int = 3) -> set[str]:
    tokens = normalize_for_duplicate(text).split()
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def ancestor_path(tweet: Tweet, tweets: dict[str, Tweet]) -> tuple[list[Tweet], str | None]:
    """Return oldest-first ancestors and an exclusion reason for malformed graphs."""
    path: list[Tweet] = []
    seen = {tweet.tweet_id}
    current = tweet
    while current.in_response_to:
        parent_id = current.in_response_to
        if parent_id in seen:
            return [], "cycle"
        parent = tweets.get(parent_id)
        if parent is None:
            return path, "missing_parent"
        if parent.created_at > current.created_at:
            return [], "future_parent"
        seen.add(parent_id)
        path.append(parent)
        current = parent
    path.reverse()
    return path, None


def connected_component_id(tweet: Tweet, tweets: dict[str, Tweet]) -> str:
    path, error = ancestor_path(tweet, tweets)
    if error == "cycle":
        return f"invalid:{tweet.tweet_id}"
    return path[0].tweet_id if path else tweet.tweet_id


def first_brand_reply(tweet: Tweet, tweets: dict[str, Tweet], brand: str) -> Tweet | None:
    candidates = [tweets[item] for item in tweet.response_ids if item in tweets]
    replies = [item for item in candidates if not item.inbound and item.author_id == brand]
    return min(replies, key=lambda item: item.created_at, default=None)


def build_episodes(tweets: dict[str, Tweet], brand: str) -> tuple[list[Episode], Counter[str]]:
    episodes: list[Episode] = []
    exclusions: Counter[str] = Counter()
    seen_components: set[str] = set()
    for tweet in sorted(tweets.values(), key=lambda item: item.created_at):
        if not tweet.inbound:
            continue
        path, error = ancestor_path(tweet, tweets)
        if error in {"cycle", "future_parent"}:
            exclusions[error] += 1
            continue
        if not any(item.author_id == brand for item in path):
            exclusions["not_addressed_to_brand"] += 1
            continue
        component_id = connected_component_id(tweet, tweets)
        if component_id in seen_components:
            exclusions["additional_message_in_component"] += 1
            continue
        seen_components.add(component_id)
        context = tuple(
            {
                "message_id": item.tweet_id,
                "author_role": "brand" if item.author_id == brand else "customer",
                "created_at": item.created_at.isoformat(),
                "text": scrub_text(item.text, brand),
            }
            for item in path
        )
        reply = first_brand_reply(tweet, tweets, brand)
        episodes.append(
            Episode(
                message_id=tweet.tweet_id,
                component_id=component_id,
                created_at=tweet.created_at.isoformat(),
                brand=brand,
                customer_text=scrub_text(tweet.text, brand),
                context=context,
                historical_reply_id=reply.tweet_id if reply else None,
                historical_reply_text=scrub_text(reply.text, brand) if reply else None,
            )
        )
    return episodes, exclusions


def time_split(episodes: list[Episode], train_fraction: float = 0.6, dev_fraction: float = 0.2) -> dict[str, list[Episode]]:
    if not 0 < train_fraction < train_fraction + dev_fraction < 1:
        raise ValueError("fractions must define three non-empty chronological partitions")
    ordered = sorted(episodes, key=lambda item: item.created_at)
    first = int(len(ordered) * train_fraction)
    second = int(len(ordered) * (train_fraction + dev_fraction))
    return {"train": ordered[:first], "dev": ordered[first:second], "test_pool": ordered[second:]}


def fingerprint(episode: Episode) -> str:
    payload = f"{normalize_for_duplicate(episode.customer_text)}\n{json.dumps(episode.context, sort_keys=True)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_jsonl(records: Iterable[object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record) if hasattr(record, "__dataclass_fields__") else record) + "\n")


def load_csv(path: Path) -> tuple[dict[str, Tweet], Counter[str]]:
    tweets: dict[str, Tweet] = {}
    exclusions: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                tweet = parse_tweet(row)
            except (KeyError, ValueError):
                exclusions["invalid_row"] += 1
                continue
            if tweet.tweet_id in tweets:
                exclusions["duplicate_tweet_id"] += 1
                continue
            tweets[tweet.tweet_id] = tweet
    return tweets, exclusions


def shortlist_brands(path: Path, limit: int = 15) -> list[dict[str, object]]:
    """Stream the raw CSV and rank candidate support handles before model work.

    An outbound row identifies the authoring brand. This is intentionally a
    volume shortlist, not a claim about quality or resolution rate; those are
    assessed in the documented early-window manual review.
    """
    outbound_counts: Counter[str] = Counter()
    informative_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if parse_bool(row.get("inbound")):
                continue
            author = (row.get("author_id") or "").strip()
            if not author:
                continue
            outbound_counts[author] += 1
            text = (row.get("text") or "").lower()
            if len(text) >= 80 and not re.search(r"\b(?:dm|pm|private message)\b", text):
                informative_counts[author] += 1
    return [
        {
            "brand": brand,
            "outbound_messages": count,
            "rough_informative_share": round(informative_counts[brand] / count, 3),
        }
        for brand, count in outbound_counts.most_common(limit)
    ]


def direct_brand_examples(path: Path, brand: str) -> list[dict[str, str | None]]:
    """Return direct customer/brand exchanges using a two-pass bounded scan.

    This is deliberately narrower than final thread reconstruction. It is used
    only for choosing a brand, where direct exchanges make manual review faster
    and avoid loading the three-million-row CSV into Python objects.
    """
    brand_rows: dict[str, Tweet] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("author_id") == brand and not parse_bool(row.get("inbound")):
                item = parse_tweet(row)
                brand_rows[item.tweet_id] = item
    brand_ids = set(brand_rows)
    examples: list[dict[str, str | None]] = []
    seen_customer_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not parse_bool(row.get("inbound")):
                continue
            item = parse_tweet(row)
            reply_ids = set(item.response_ids) & brand_ids
            parent_is_brand = item.in_response_to in brand_ids
            if not reply_ids and not parent_is_brand:
                continue
            if item.tweet_id in seen_customer_ids:
                continue
            seen_customer_ids.add(item.tweet_id)
            reply_id = min(reply_ids, key=lambda value: brand_rows[value].created_at, default=None)
            parent = brand_rows.get(item.in_response_to or "")
            examples.append(
                {
                    "customer_id": item.tweet_id,
                    "component_id": item.in_response_to if parent_is_brand else item.tweet_id,
                    "created_at": item.created_at.isoformat(),
                    "customer_text": scrub_text(item.text, brand),
                    "prior_brand_text": scrub_text(parent.text, brand) if parent else None,
                    "brand_reply_id": reply_id,
                    "brand_reply_text": scrub_text(brand_rows[reply_id].text, brand) if reply_id else None,
                }
            )
    return examples


def write_brand_review_samples(path: Path, brands: list[str], sample_size: int, seed: int) -> None:
    """Write fixed early-window samples for the manual brand-selection review."""
    output_dir = ROOT / "reports/brand_review_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for brand in brands:
        examples = sorted(direct_brand_examples(path, brand), key=lambda item: str(item["created_at"]))
        early_count = max(sample_size, int(len(examples) * 0.2))
        early = examples[:early_count]
        chosen = random.Random(f"{seed}:{brand}").sample(early, k=min(sample_size, len(early)))
        write_jsonl(chosen, output_dir / f"{brand}.jsonl")


def fetch() -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "thoughtvector/customer-support-on-twitter", "-p", str(RAW_CSV.parent), "--unzip"],
        check=True,
    )


def build(path: Path, brand: str) -> None:
    examples = direct_brand_examples(path, brand)
    episode_exclusions: Counter[str] = Counter()
    seen_components: set[str] = set()
    episodes: list[Episode] = []
    for example in sorted(examples, key=lambda item: str(item["created_at"])):
        component_id = str(example["component_id"])
        if component_id in seen_components:
            episode_exclusions["additional_message_in_component"] += 1
            continue
        seen_components.add(component_id)
        prior_brand_text = example["prior_brand_text"]
        context = (
            (
                {
                    "message_id": component_id,
                    "author_role": "brand",
                    "created_at": "unknown",
                    "text": str(prior_brand_text),
                },
            )
            if prior_brand_text
            else ()
        )
        episodes.append(
            Episode(
                message_id=str(example["customer_id"]),
                component_id=component_id,
                created_at=str(example["created_at"]),
                brand=brand,
                customer_text=str(example["customer_text"]),
                context=context,
                historical_reply_id=example["brand_reply_id"],
                historical_reply_text=example["brand_reply_text"],
            )
        )
    partitions = time_split(episodes)
    for name, records in partitions.items():
        write_jsonl(records, INTERIM / f"{brand}_{name}.jsonl")
    manifest = {
        "brand": brand,
        "source": str(path),
        "direct_brand_examples": len(examples),
        "episodes": len(episodes),
        "partition_counts": {name: len(records) for name, records in partitions.items()},
        "episode_exclusions": dict(episode_exclusions),
    }
    (INTERIM / f"{brand}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_shortlist(path: Path, limit: int) -> None:
    candidates = shortlist_brands(path, limit)
    output = ROOT / "reports/brand_selection.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Brand shortlist\n",
        "This preliminary ranking uses outbound-message volume. The `rough_informative_share` is a coarse text heuristic for prioritising the manual early-window review; it is not a quality or resolution metric. No agent result informed this shortlist.\n",
        "| Brand | Outbound messages | Rough informative share |",
        "|---|---:|---:|",
    ]
    rows.extend(
        f"| {item['brand']} | {item['outbound_messages']:,} | {item['rough_informative_share']:.1%} |"
        for item in candidates
    )
    rows.extend(
        [
            "",
            "Selection remains pending manual review of 30 early-window conversations per shortlisted candidate. The final selection will favour reusable public guidance, viable volume, diverse intents and interpretable escalation cases.",
        ]
    )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    shortlist_parser = commands.add_parser("shortlist")
    shortlist_parser.add_argument("--input", type=Path, default=RAW_CSV)
    shortlist_parser.add_argument("--limit", type=int, default=15)
    review_parser = commands.add_parser("review-samples")
    review_parser.add_argument("--input", type=Path, default=RAW_CSV)
    review_parser.add_argument("--brands", nargs="+", required=True)
    review_parser.add_argument("--sample-size", type=int, default=30)
    review_parser.add_argument("--seed", type=int, default=20260909)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--input", type=Path, default=RAW_CSV)
    build_parser.add_argument("--brand", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    elif args.command == "shortlist":
        write_shortlist(args.input, args.limit)
    elif args.command == "review-samples":
        write_brand_review_samples(args.input, args.brands, args.sample_size, args.seed)
    else:
        build(args.input, args.brand)


if __name__ == "__main__":
    main()
