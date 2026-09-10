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
2. `src/preannotate.py` drafts an intent, an escalate/auto call, and a
   one-line reason for every candidate, using a model from a third family
   (Qwen via Groq) distinct from both the generator (Groq gpt-oss) and the
   judge (Gemini).
3. Ashraf reviewed all 250 drafts in a spreadsheet export
   (`data/golden/golden_labelling.csv`) against the codebook and confirmed
   every one — 0/250 overrides on intent, escalation call, and reason.

**What "0/250 overrides" does and doesn't mean:** it means the reviewer
agreed with the model's draft on every item, not that the labels were
written independently from scratch. `label_source: "ai_drafted_human_reviewed"`
is stamped on every record for exactly this reason — the wording of the
`escalate_reason` field, in particular, is the model's, confirmed rather
than authored, and that distinction matters if asked to defend any specific
label's phrasing. The intent and escalate/auto *decision* is a human
judgment call in every case; the review was a single spreadsheet pass, not
a slower per-item confirmation, and a 0% override rate can't by itself
distinguish "the drafts were accurate" from "the review was shallow" — both
are plausible, and there's no independent measurement in this dataset that
separates them.

## What is misleading about my headline number? (draft — will be finished once evaluation runs)

Points already known to belong here, ahead of the full write-up:

- **The 0/250 override rate is not independent evidence of label quality.**
  See above — it's confirmed-by-review, not written-from-scratch, and there's
  no held-out subset in this dataset that measures whether the draft
  suggestions biased the reviewer.
- **A single reviewer, no second rater.** All 250 golden labels were
  confirmed by one person, so there's no inter-annotator agreement number.
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
