# Hiver support-agent take-home

A retrieval-grounded support-drafting agent for **SpotifyCares**, built on the
Kaggle *Customer Support on Twitter* dataset. For each inbound customer
message it returns an intent, a reply draft grounded in that brand's
historical resolutions, and `auto` or `escalate` with a stated reason.

Historical replies are treated as examples of wording and general public
guidance — not proof an issue was resolved, and not current policy. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and
[DECISIONS.md](DECISIONS.md) for the non-obvious calls made along the way.

**Status: in progress.** Data pipeline, retrieval, agent, guardrails,
baselines, taxonomy, and the golden-set candidate pool are built and tested.
The golden-set labelling session, the end-to-end eval run, judge-agreement
study, and `REPORT.md` are not yet done — those require a human labelling
pass (~3-4 hours) that hasn't happened yet. `make reproduce` is not yet
runnable to completion; run `make test` for what's currently verifiable.

## Setup

```bash
uv sync --python 3.12 --all-groups
```

Requires `GROQ_API_KEY` (generator + pre-annotator) and `GEMINI_API_KEY`
(judge) as environment variables for any step that calls a live model.
Cached LLM outputs live in `artifacts/llm_cache.jsonl` so re-runs don't
re-call the API.

## What's built and tested

```bash
make test          # 27 tests: thread reconstruction, redaction, policy,
                    # retrieval, metrics, cost model, baselines, sampling
```

- `src/data.py` — brand shortlisting, thread reconstruction, PII redaction,
  chronological train/dev/test_pool split
- `src/retrieve.py` — BM25 retrieval over historical resolved threads
- `src/agent.py` — structured-output generation + citation validation
- `src/policy.py` — deterministic guardrails (unsupported commitments,
  sensitive-data requests, private-handoff detection)
- `src/taxonomy.py` — TF-IDF/KMeans clustering used to *discover* candidate
  intents (see `data/golden/codebook.md` for the hand-written taxonomy)
- `src/baselines.py` — trivial baseline (majority intent, canned reply,
  fixed routing) and simple baseline (TF-IDF+LogReg intent, BM25-copy reply)
- `src/sampling.py` — stratified golden-set candidate sampling from the
  held-out `test_pool` split (200 candidates, 10 intents, hard cases
  oversampled)
- `src/preannotate.py` — pre-annotator label suggestions for the labelling
  session (Qwen family via Groq — a third model family, distinct from both
  the generator and the judge)
- `src/eval/metrics.py`, `risk_coverage.py`, `judge.py`, `judge_agreement.py`
  — intent/routing metrics with bootstrap CIs, the coverage-at-fixed-safety
  headline calculation, the LLM judge + two decoy judges, and the
  judge-vs-human agreement study

## What's next (requires a human)

```bash
make candidates     # already run — 200 candidates in data/golden/golden_candidates.jsonl
make preannotate    # already run — suggestions added for 150/200 candidates
make label          # ~2-3 hours: intent + escalation labelling in the terminal TUI
make rate-replies   # ~1 hour, after an initial eval run produces drafts to rate
```

Then `make live` runs the full pipeline against the real APIs and populates
the cache; `make reproduce` replays from that cache with no network or keys
and regenerates every number that will appear in `REPORT.md`.

## Repo map

```
src/            pipeline code (see file list above)
src/eval/       metrics, judge, judge-agreement study, risk-coverage
data/interim/   SpotifyCares_{train,dev,test_pool}.jsonl (chronological split)
data/golden/    codebook.md, golden_candidates.jsonl, (golden_v1.jsonl pending)
reports/        brand_selection.md, brand_review_samples/, taxonomy_clusters.txt
artifacts/      cached LLM outputs, results.json (pending)
tests/          27 tests covering every module above
```
