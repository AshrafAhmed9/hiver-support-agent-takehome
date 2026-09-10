# Evaluation report — work in progress

No end-to-end performance result has been measured yet. This document
records status and methodology so far; it is not the completed assignment
report. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full
design and [DECISIONS.md](DECISIONS.md) for why things were built this way.

## Golden-set status

`data/golden/golden_v1.jsonl` currently has 150 of 200 items:

- **150 items** were sampled with a pre-annotator suggestion shown
  (intent, escalate/auto, one-line reason — all from a third model family,
  distinct from both the generator and the judge). Ashraf reviewed the full
  set in a spreadsheet export and confirmed agreement with every suggestion;
  those 150 were then written directly rather than confirmed one item at a
  time in the terminal labelling tool, so they're tagged `bulk_accepted:
  true` in both `golden_v1.jsonl` and `labeling_log.jsonl`. See
  [DECISIONS.md #5](DECISIONS.md) for the full rationale.
- **50 items** were sampled with no suggestion shown at all (blind), and
  still need independent labelling via `label_tui.py --blind-only`. These
  are the only source of a genuinely independent override/anchoring signal
  in this dataset — the 150 bulk-accepted items can't tell you anything
  about how well the pre-annotator's suggestions hold up against a human
  working with no hint.

## What is misleading about my headline number? (draft — will be finished once evaluation runs)

Points already known to belong here, ahead of the full write-up:

- **The override rate on the 150 bulk-accepted items (0/150) is not, by
  itself, strong evidence of pre-annotator accuracy.** It reflects one pass
  of spreadsheet review rather than the slower per-item confirmation the
  TUI produces. The 50 blind items' independent labels are the number to
  weight more heavily when judging how good the suggestions actually were.
- **A single annotator, no second rater.** All 200 golden labels — bulk-
  accepted or blind — come from one person. No inter-annotator agreement
  number exists for this dataset.
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

- The 50 blind golden labels (`label_tui.py --blind-only`)
- End-to-end generation + judging run against both baselines
- Judge-human agreement study (Spearman/weighted-kappa per dimension, decoy
  judges as controls, position-bias check) — needs ~80 human reply ratings,
  which need the eval run's drafts first
- Bootstrap CIs on every headline metric
- Risk-coverage curve and the cost-ratio sensitivity sweep
- Five real failure modes with examples and frequency counts
- One-more-week section
