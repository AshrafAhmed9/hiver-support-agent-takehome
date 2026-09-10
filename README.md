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
Reference labels are AI-assigned with per-record provenance in `data/labels/`;
they are not human labels. The end-to-end eval run and judge-agreement
study are not yet done. No judge–human agreement has been measured. `make reproduce` is not yet
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
make test          # thread reconstruction, redaction, policy,
                    # retrieval, metrics, cost model, baselines, sampling
```

- `src/data.py` — brand shortlisting, thread reconstruction, PII redaction,
  chronological train/dev/test_pool split
- `src/retrieve.py` — BM25 retrieval over historical resolved threads
- `src/agent.py` — structured-output generation + citation validation
- `src/policy.py` — deterministic guardrails (unsupported commitments,
  sensitive-data requests, private-handoff detection)
- `src/taxonomy.py` — TF-IDF/KMeans clustering used to *discover* candidate
  intents (see `data/golden/codebook.md` for the AI-authored working taxonomy)
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

## AI labeling and remaining evaluation work

The authorized labeling workflow is:

```bash
uv run python -m src.ai_label
```

It creates 150 training, 60 development and 200 evaluation reference labels with
AI provenance, routing rationales and uncertainty flags. See
[labeling notes](data/labels/README.md). The current evaluation candidates are a
stratified challenge sample, so results must not be presented as inbound-volume
estimates. The following older commands remain optional human-review utilities;
they have not produced human labels or ratings:

```bash
make candidates     # already run — 200 candidates in data/golden/golden_candidates.jsonl
make preannotate    # already run — suggestions added for 150/200 candidates
make label          # optional actual human labeling in the terminal TUI
make rate-replies   # optional actual human ratings, after draft generation
```

The end-to-end evaluation runner is still an unfinished integration point.
`make live` and `make reproduce` do not yet produce evaluation results.

## Repo map

```
src/            pipeline code (see file list above)
src/eval/       metrics, judge, judge-agreement study, risk-coverage
data/interim/   SpotifyCares_{train,dev,test_pool}.jsonl (chronological split)
data/golden/    codebook.md, golden_candidates.jsonl, (golden_v1.jsonl pending)
data/labels/    AI train/dev/challenge labels, per-item provenance, manifest and notes
reports/        brand_selection.md, brand_review_samples/, taxonomy_clusters.txt
artifacts/      cached LLM outputs, results.json (pending)
tests/          focused tests, including AI label provenance and validation
```
