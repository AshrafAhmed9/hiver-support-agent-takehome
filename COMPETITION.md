# Hiver SDE Intern — Take-Home Assignment

## Headline (written before building)
> A retrieval-grounded support agent for SpotifyCares that classifies intent,
> drafts a grounded reply, and routes auto/escalate — evaluated honestly
> against two baselines, with the judge itself checked against 80 blind
> human ratings.
- Number it rests on: coverage-at-95%-safety = 1.7% (5/60... actually 1/60
  at threshold 0.88) — a deliberately unflattering headline, reported
  straight rather than smoothed. See REPORT.md's misleading-number section.
- Still accurate at freeze? Not yet frozen — REPORT.md is stale/incomplete
  relative to the actual last eval run. See below.

## Facts
- Competition type: take-home / hiring assignment.
- Deadline: not stated by user; treat submission as imminent.
- Judged artifact: the repo + report, read by a human reviewer who will
  also ask the candidate to explain and modify the code live.
- No field/entrant count — not a leaderboard competition. Field-counting,
  prize-bucket, and idea-generation sections of the competition skill are
  N/A here; take-home carryover rules apply (see skill's take-home note).
- Rule confirmed from the brief: "You may use AI coding assistants freely...
  cite anything you borrowed." This governs code, not the golden-set labels
  or blind human ratings, whose entire evidentiary value is being
  independently human — see DECISIONS.md #5 and reply_rating_csv.py.

## Deliverables checklist (from the brief, verbatim requirements)

| Deliverable | Status | Notes |
|---|---|---|
| Runnable repo, reproducible in <15 min | ✅ Fixed this session | `make reproduce` was broken (defaulted to sample-size 250 against a cache that only covers the 60 actually run live) — now pinned to 60, runs in ~3s |
| Golden eval set, 150–250 hand-labelled, sampling+labelling note | ✅ | 250 items, hand-labelled by Ashraf, documented in REPORT.md + codebook.md + DECISIONS.md #5 |
| Eval harness: automated metrics + LLM-judge rubric | ✅ | metrics.py, judge.py, risk_coverage.py all wired and running |
| Judge-vs-human agreement evidence | ⚠️ Fixed 2 real bugs this session | See "Bugs fixed" below — numbers were wrong, now correct but still show a **weak** judge (this is a legitimate, reportable finding, not a blocker) |
| Report: problem framing | ⚠️ Present but thin | in IMPLEMENTATION_PLAN.md, not consolidated into REPORT.md |
| Report: results vs ≥2 baselines | ❌ Not written up | Numbers exist in artifacts/results.json (agent vs simple_bm25/tfidf vs trivial) but REPORT.md doesn't present them |
| Report: top-5 failure modes w/ real examples | ❌ Missing entirely | REPORT.md's "Remaining work" list still says this is TODO |
| Report: "what is misleading about my headline number" | ⚠️ Draft, incomplete | Has 3 good points but marked "draft — will be finished once evaluation runs" — evaluation HAS run, section wasn't updated |
| Report: next steps / one more week | ❌ Missing | Listed as TODO in "Remaining work" |
| Decision log, 10–15 non-obvious decisions | ✅ | DECISIONS.md has 13, well-written with tradeoffs |

## Bugs fixed this session (all in the judge-agreement path)

1. **judge_agreement.py matched on bare `item_id`.** `judge_scores.jsonl`
   holds one row per (item, system) sharing an id; the rating pool suffixes
   `_simple` onto the simple-baseline's id to stay unique in the flat CSV.
   Matching on raw item_id silently collapsed agent+simple judge scores
   (last-write-wins → always simple) and dropped every simple-baseline
   human rating as unmatched. `n_items` was 40/80; real_judge Spearman was
   negative on 2 of 4 dimensions. Fixed to match on (base_id, system).
   Real numbers post-fix: weak positive agreement (ρ 0.0–0.29), still not
   a strong judge, but no longer distorted by the bug.
2. **position_bias_flip_rate had the same collapse bug** — the swapped
   (agent-only) re-judged scores were compared against whichever system's
   score was last written per id, i.e. almost always the wrong reply.
   Reported flip rate went from 56.7% → 26.7% after fixing.
3. **`make reproduce` didn't match how the live run was invoked** —
   defaulted to the full 250-item golden set; the actual live run (and its
   cache) only covers 60 due to Groq's daily token quota. Fixed to pass
   `--sample-size 60` consistently in both `reproduce` and `live` targets.

These are exactly the kind of thing a reviewer who runs the repo themselves
will hit immediately (`make reproduce` failing outright is a first-impression
killer), and exactly the kind of silent-wrong-number bug the brief's mandatory
"what is misleading" section is designed to catch — except this one was in
the harness itself, not the model.

## What's actually blocking a strong submission right now

**REPORT.md is stale, not missing infrastructure.** The eval has genuinely
been run (artifacts/results.json is real, complete, and reproducible), but
REPORT.md still opens with "No end-to-end performance result has been
measured yet" and lists the entire results/failure-analysis/next-steps
content as "Remaining work." A reviewer reading REPORT.md as written would
conclude the assignment is unfinished, when the actual gap is a report that
wasn't updated after the run. This is the single highest-priority fix.

**README's status line is also stale** ("Status: in progress... `make
reproduce` is not yet runnable to completion") — now false on both counts.

**The headline number is genuinely bad (1.7% coverage) — report it, don't
soften it.** At the safety target (95% of auto-replies must be safe), the
system can only confidently auto-handle 1 of 60 golden items. That's either
a legitimate, honestly-reported finding about how conservative the guardrail
+ confidence threshold combination is, or a sign the threshold/features need
another pass. Given the brief rewards honesty over polish, writing this up
straight (with the cost-ratio sensitivity table already computed) is the
right move, not chasing a bigger number.

## Claims
| Claim | Verifiable proof | Built? | Where a reviewer sees it |
|---|---|---|---|
| Agent beats both baselines on intent accuracy | results.json: agent 63.3% [51.7,75.0] vs simple 38.3% vs trivial 0% | ✅ computed | Not yet in REPORT.md |
| Guardrails are a hard gate, not a suggestion | policy.py + DECISIONS.md #8 | ✅ | code + decision log |
| Judge validated against independent human ratings | judge_agreement_report.json, now bug-fixed | ✅ | Not yet in REPORT.md |
| Golden labels are genuinely human, not model-assisted | label_source: "human" on all 250, no pre-annotator in the pipeline | ✅ | README, REPORT.md, DECISIONS.md #5 |
| 250-item golden set, only 60 evaluated live | disclosed in run_eval.py docstring + results.json's n_golden field | ✅ but under-surfaced | needs to be stated up top in REPORT.md, not just in a docstring |

## Freeze checklist (take-home subset)
0. Type identified: take-home. Field/prize items N/A.
1–2. N/A (no field to count).
3. Tool limitations: Groq quota forced a 60-item subsample — disclosed, and the brief explicitly permits subsampling. Not routed around by faking a full run.
4. One proof per core claim: mostly yes (metrics computed), not yet all surfaced in the report.
5. System's limits shown: yes in principle (1.7% coverage, honest low number) — but not yet written up where a reviewer will read it.
6. Localization: N/A for this domain.
7. Real inputs: yes, real Twitter support data throughout.
8. One memorable number visible early: not yet — REPORT.md's stale opening line is the first thing a reviewer reads.
9. N/A — no scored sponsor tech.
10. Core claim demonstrated twice: agent vs. both baselines, yes, once results are written up.
11. Every claim visible in the judged artifact, not just repo: **not yet** — this is the main open item.
12. N/A.
13. Headline still accurate: yes, just not written down yet.
14. Freeze: not yet — REPORT.md needs the actual writing pass next.
15. Whole path rehearsed: `make reproduce` now genuinely works end to end.
16. N/A.
17. Submission surface: README is close, needs the status-line fix.

## Overruled concerns
None — no direction was overruled this session; three real bugs and one
broken repro path were found and fixed, and the state above reflects them
honestly rather than as fully resolved.

## Post-results review
N/A until submitted.
