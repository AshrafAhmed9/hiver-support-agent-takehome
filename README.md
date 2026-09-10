# Hiver support-agent take-home

A retrieval-grounded support-drafting agent for **SpotifyCares**, built on the
Kaggle *Customer Support on Twitter* dataset. For each inbound customer
message it returns an intent, a reply draft grounded in that brand's
historical resolutions, and `auto` or `escalate` with a stated reason.

Historical replies are treated as examples of wording and general public
guidance, not proof an issue was resolved, and not current policy. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and
[DECISIONS.md](DECISIONS.md) for the non-obvious calls made along the way.

**Status: eval run complete on a 60-item subsample of the 250-item golden
set** (Groq's daily token quota doesn't stretch to 250 live generations in
one sitting; see DECISIONS.md and REPORT.md). `make reproduce` replays that
run from the committed cache in a few seconds, no network or keys required.

## Setup

```bash
uv sync --python 3.12 --all-groups
```

Requires `GROQ_API_KEY` (generator + judge) as an environment variable for
any step that calls a live model.
Cached LLM outputs live in `artifacts/llm_cache.jsonl` so re-runs don't
re-call the API.

## What's built and tested

```bash
make test          # thread reconstruction, redaction, policy,
                    # retrieval, metrics, cost model, baselines, sampling
```

- `src/data.py`: brand shortlisting, thread reconstruction, PII redaction,
  chronological train/dev/test_pool split
- `src/retrieve.py`: BM25 retrieval over historical resolved threads
- `src/agent.py`: structured-output generation + citation validation
- `src/policy.py`: deterministic guardrails (unsupported commitments,
  sensitive-data requests, private-handoff detection)
- `src/taxonomy.py`: TF-IDF/KMeans clustering used to *discover* candidate
  intents (see `data/golden/codebook.md` for the hand-written taxonomy)
- `src/baselines.py`: trivial baseline (majority intent, canned reply,
  fixed routing) and simple baseline (TF-IDF+LogReg intent, BM25-copy reply)
- `src/sampling.py`: stratified golden-set candidate sampling from the
  held-out `test_pool` split (250 candidates, 10 intents, hard cases
  oversampled)
- `src/golden_csv.py`: the spreadsheet labelling workflow (see below)
- `src/eval/metrics.py`, `risk_coverage.py`, `judge.py`, `judge_agreement.py`:
  intent/routing metrics with bootstrap CIs, the coverage-at-fixed-safety
  headline calculation, the LLM judge + two decoy judges, and the
  judge-vs-human agreement study

## Golden-set labelling status

`data/golden/golden_v1.jsonl` has all 250 items. How they were made, stated
plainly:

1. 250 candidates sampled from the held-out `test_pool` split (`src/sampling.py`)
2. Ashraf labelled all 250 by hand in a spreadsheet against the codebook:
   intent, escalate/auto call, and a one-line reason each (`src/golden_csv.py`)

Every record carries `label_source: "human"`. See
[REPORT.md](REPORT.md)'s golden-set section for the known limits of a
single-annotator set.

`make live` runs the full pipeline against the real APIs; `make reproduce`
replays from `artifacts/llm_cache.jsonl` with no network or keys and
regenerates every number in `REPORT.md`.

## Repo map

```
src/            pipeline code (see file list above)
src/eval/       metrics, judge, judge-agreement study, risk-coverage
data/interim/   SpotifyCares_{train,dev,test_pool}.jsonl (chronological split)
data/golden/    codebook.md, golden_candidates.jsonl, golden_labelling.csv,
                golden_v1.jsonl (250/250)
reports/        brand_selection.md, brand_review_samples/, taxonomy_clusters.txt
artifacts/      cached LLM outputs, results.json, judge_agreement_report.json
tests/          tests covering every module above
```
