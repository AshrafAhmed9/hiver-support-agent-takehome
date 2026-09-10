# Evaluation report — work in progress

No end-to-end performance result has been measured yet. This document
records status and methodology so far; it is not the completed assignment
report. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full
design and [DECISIONS.md](DECISIONS.md) for why things were built this way.

## Golden-set status

`data/golden/golden_v1.jsonl` has all 250 items. Here's exactly how they
were produced, stated plainly rather than glossed over:

1. `src/sampling.py` selects 250 candidate messages from the held-out
   `test_pool` split, stratified across the 10 intents with rare/ambiguous
   cases oversampled (see [the codebook](data/golden/codebook.md)).
2. Ashraf labelled all 250 by hand in a spreadsheet export
   (`data/golden/golden_labelling.csv`) against the codebook — intent, the
   escalate/auto call, and a one-line reason for each.

Every record carries `label_source: "human"`. The labels, including the
wording of every `escalate_reason`, are the annotator's own.

## What is misleading about my headline number? (draft — will be finished once evaluation runs)

Points already known to belong here, ahead of the full write-up:

- **A single annotator, no second rater.** All 250 golden labels were
  written by one person, so there's no inter-annotator agreement number and
  no independent check on systematic bias in how the codebook was applied.
- **The golden set is a stratified challenge sample, not a volume-weighted
  sample.** Rare intents and ambiguous cases were deliberately oversampled
  so the system gets stress-tested; this means the golden set's intent
  distribution does not represent ordinary inbound message volume, and any
  future accuracy number should not be read as "this is what happens to a
  random customer message."
- **Twitter support is not Hiver's actual product context** — public, short,
  low-stakes messages, versus the longer, private, higher-stakes email
  threads Hiver's customers actually run through a shared inbox.

## Remaining work

- End-to-end generation + judging run against both baselines
- Judge-human agreement study (Spearman/weighted-kappa per dimension, decoy
  judges as controls, position-bias check) — needs ~80 human reply ratings,
  which need the eval run's drafts first
- Bootstrap CIs on every headline metric
- Risk-coverage curve and the cost-ratio sensitivity sweep
- Five real failure modes with examples and frequency counts
- One-more-week section
