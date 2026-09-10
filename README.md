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
150 of 200 golden labels are done (reviewed and confirmed by hand — see
DECISIONS.md #16 for exactly how); the remaining 50 blind items and the
end-to-end eval run are not yet done. `make reproduce` is not yet runnable
to completion; run `make test` for what's currently verifiable.

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
  intents (see `data/golden/codebook.md` for the hand-written taxonomy)
- `src/baselines.py` — trivial baseline (majority intent, canned reply,
  fixed routing) and simple baseline (TF-IDF+LogReg intent, BM25-copy reply)
- `src/sampling.py` — stratified golden-set candidate sampling from the
  held-out `test_pool` split (200 candidates, 10 intents, hard cases
  oversampled)
- `src/preannotate.py` — pre-annotator label suggestions for the labelling
  session (Qwen family via Groq — a third model family, distinct from both
  the generator and the judge)
- `src/label_tui.py` / `src/golden_csv.py` — the human labelling tools (see
  below)
- `src/eval/metrics.py`, `risk_coverage.py`, `judge.py`, `judge_agreement.py`
  — intent/routing metrics with bootstrap CIs, the coverage-at-fixed-safety
  headline calculation, the LLM judge + two decoy judges, and the
  judge-vs-human agreement study

## Golden-set labelling status and how to finish it

```bash
make csv-export         # already run — data/golden/golden_labelling.csv
make accept-suggested   # already run — 150/200 reviewed & confirmed by hand
make label-blind-only   # remaining: 50 items, no suggestion, ~15-20 min
```

`data/golden/golden_v1.jsonl` currently has 150 of 200 items. Every record
carries `label_source: "human"` and `human_reviewed: true`; the 150 bulk-
accepted ones additionally carry `bulk_accepted: true`, meaning they were
confirmed by reviewing the full suggested set in a spreadsheet rather than
one-by-one in the terminal tool — disclosed rather than hidden, see
DECISIONS.md #16. The remaining 50 have no suggestion at all and require
`make label-blind-only`.

Once all 200 exist, `make live` runs the full pipeline against the real
APIs; `make reproduce` replays from `artifacts/llm_cache.jsonl` with no
network or keys and regenerates every number in `REPORT.md`.

## Repo map

```
src/            pipeline code (see file list above)
src/eval/       metrics, judge, judge-agreement study, risk-coverage
data/interim/   SpotifyCares_{train,dev,test_pool}.jsonl (chronological split)
data/golden/    codebook.md, golden_candidates.jsonl, golden_labelling.csv,
                golden_v1.jsonl (150/200), labeling_log.jsonl
reports/        brand_selection.md, brand_review_samples/, taxonomy_clusters.txt
artifacts/      cached LLM outputs, results.json (pending)
tests/          tests covering every module above
```
