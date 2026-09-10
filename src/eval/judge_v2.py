"""Recalibrated judge: same rubric as judge.py, plus few-shot worked
examples pulled from a held-out slice of the human reply ratings.

Why: the judge-agreement study (judge_agreement.py, REPORT.md) found the
original zero-shot judge doesn't clearly beat a length-only decoy on 3 of
4 dimensions, and has zero rank correlation with human ratings on safety.
DECISIONS.md #14 explains why that judge model was picked under API
constraints rather than validated for quality first. This module is the
fix attempt: anchor the rubric with real examples of what a 2 vs. a 4
actually looks like on this data, instead of describing the scale in
words alone.

Calibration/holdout split, to avoid grading on the training set:
`data/golden/reply_ratings.jsonl` (80 items) is split deterministically
(seeded on RANDOM_SEED) into 20 calibration items (used to build the
few-shot examples embedded in the prompt) and 60 holdout items (the only
ones ever scored by the recalibrated judge or used to re-measure
agreement). The recalibrated judge must never be validated against an
item it was shown as an example — that would just be measuring whether it
can copy an answer it already has.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import RANDOM_SEED
from src.eval.judge import JudgeScore, _parse_json_score
from src.llm import CachedLLM

ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = ROOT / "data/golden/reply_ratings.jsonl"
POOL_PATH = ROOT / "artifacts/predictions_for_rating.jsonl"

N_CALIBRATION = 20
N_FEWSHOT_SHOWN = 6

RUBRIC_HEADER = """Score the proposed reply on 4 dimensions, each 1-5:

groundedness: does the reply only propose remedies that actually appear in the cited historical evidence?
  1 = invents a remedy not in the evidence, 3 = partially grounded, 5 = every claim is directly supported by the evidence.
resolution_helpfulness: would this plausibly resolve the customer's stated issue?
  1 = does not address the issue, 3 = partially addresses it, 5 = a customer could act on this and be done.
brand_voice: does the tone/length/style match how this brand's support team actually writes?
  1 = generic corporate boilerplate unlike the brand, 3 = plausible but somewhat off, 5 = indistinguishable from the brand's real replies.
safety: does it avoid promising something the brand can't verify from a public tweet (refunds, dates, account actions)?
  1 = makes an unsupported commitment, 3 = borderline, 5 = clearly safe.

Below are worked examples, scored by a human reviewer, showing what these
scores actually look like on real replies. Calibrate against them,
especially safety: a reply that merely offers to look into something is
not automatically a 5, and a reply that is polite but generic is not
automatically a 5 on resolution_helpfulness either.
"""

RUBRIC_FOOTER = """
Now score this one the same way:

Customer message:
{customer_text}

Evidence (historical resolved threads for this brand):
{evidence}

Proposed reply:
{reply}

Respond with ONLY a JSON object: {{"groundedness": <1-5>, "resolution_helpfulness": <1-5>, "brand_voice": <1-5>, "safety": <1-5>, "rationale": "<one sentence>"}}"""


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def calibration_holdout_split(
    human_ratings: list[dict], seed: int = RANDOM_SEED, n_calibration: int = N_CALIBRATION
) -> tuple[list[str], list[str]]:
    """Deterministic, seeded split of item_ids into calibration/holdout.
    Same seed every run — this is a fixed split, not resampled per call, so
    the holdout set is stable and auditable."""
    ids = sorted(r["item_id"] for r in human_ratings)  # sort first: dict/file order isn't guaranteed stable
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    calibration_ids = shuffled[:n_calibration]
    holdout_ids = shuffled[n_calibration:]
    return calibration_ids, holdout_ids


def _select_fewshot(
    calibration_ratings: list[dict], reply_pool: dict[str, dict], k: int = N_FEWSHOT_SHOWN
) -> list[dict]:
    """Picks k examples spanning the score range (low/mid/high mean score),
    not just the first k — a judge shown only high-scoring examples learns
    nothing about what a 2 looks like."""
    scored = []
    for r in calibration_ratings:
        item = reply_pool.get(r["item_id"])
        if item is None:
            continue
        mean = sum(r["scores"].values()) / len(r["scores"])
        scored.append((mean, r, item))
    scored.sort(key=lambda t: t[0])
    if len(scored) <= k:
        return [{"rating": r, "item": item} for _, r, item in scored]
    # spread indices evenly across the sorted-by-mean list
    step = (len(scored) - 1) / (k - 1)
    picked_idx = sorted({round(i * step) for i in range(k)})
    return [{"rating": scored[i][1], "item": scored[i][2]} for i in picked_idx]


def _format_fewshot_block(examples: list[dict]) -> str:
    blocks = []
    for ex in examples:
        r, item = ex["rating"], ex["item"]
        s = r["scores"]
        blocks.append(
            f"---\nCustomer message:\n{item['customer_text']}\n\n"
            f"Proposed reply:\n{item['reply_draft']}\n\n"
            f"Human scores: groundedness={s['groundedness']}, "
            f"resolution_helpfulness={s['resolution_helpfulness']}, "
            f"brand_voice={s['brand_voice']}, safety={s['safety']}\n---"
        )
    return "\n\n".join(blocks)


def build_rubric_v2() -> str:
    """Builds the calibrated rubric string once, from the calibration split
    of the real human ratings. Raises if the ratings/pool files don't exist
    yet — this can only run after src.reply_rating_csv import has produced
    data/golden/reply_ratings.jsonl."""
    human = _load_jsonl(RATINGS_PATH)
    if not human:
        raise ValueError(f"No human reply ratings at {RATINGS_PATH} — run src.reply_rating_csv import first.")
    pool = {p["item_id"]: p for p in _load_jsonl(POOL_PATH)}
    calibration_ids, _ = calibration_holdout_split(human)
    calibration_ratings = [r for r in human if r["item_id"] in set(calibration_ids)]
    examples = _select_fewshot(calibration_ratings, pool)
    return RUBRIC_HEADER + "\n" + _format_fewshot_block(examples) + "\n" + RUBRIC_FOOTER


def judge_reply_v2(llm: CachedLLM, rubric_v2: str, customer_text: str, evidence: list[str], reply: str, max_attempts: int = 3) -> JudgeScore:
    prompt = rubric_v2.format(
        customer_text=customer_text,
        evidence="\n".join(f"- {e}" for e in evidence) or "(none retrieved)",
        reply=reply,
    )
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        params = {"temperature": 0.0} if attempt == 0 else {"temperature": 0.0, "_retry_nonce": attempt}
        response = llm.generate(prompt, params)
        try:
            return _parse_json_score(response.text)
        except (ValueError, KeyError, TypeError) as exc:
            last_exc = exc
    raise ValueError(f"judge_reply_v2: could not parse a valid score after {max_attempts} attempts: {last_exc}")

