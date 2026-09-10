# Evaluation report

A support-drafting agent for **SpotifyCares**: given an inbound customer
message, it returns an intent, a reply draft grounded in how this brand has
historically handled similar issues, and an `auto`/`escalate` decision with
a stated reason. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for
the full design and [DECISIONS.md](DECISIONS.md) for why things were built
the way they were.

Everything below is computed from a **60-item subsample of the 250-item
golden set** (`make reproduce`, ~3 seconds, no network or API keys). The
subsample is a real constraint, not a cherry-pick (see the misleading-number
section), and the brief explicitly allows it: "we will not run your code on
the full dataset, a subsample is expected and encouraged."

## What "good" means for this brand, and what I chose not to build

SpotifyCares' Twitter support is mostly short, public, low-context messages:
login problems, billing confusion, "why isn't this song here," feature
requests, and a fair amount of noise that isn't a support request at all
("thanks!", song lyrics, unrelated mentions). Nobody is putting their
account password in a tweet. That shapes what "good" means here:

- **A safe auto-reply is one that doesn't need account access and doesn't
  promise anything the agent can't verify.** "Try restarting the app" is
  fine to send blind. "Your refund is processed" is not: the agent has no
  way to check that, so it can't say it.
- **Historical replies are examples of wording, not proof anything was
  resolved.** The dataset shows what Spotify's support team said, not
  whether it worked. The agent is grounded in tone and common remedies, not
  in "this fixed it for someone else."
- **Escalation is the safe default, not a failure state.** A system that
  escalates too much is annoying. A system that auto-answers something it
  shouldn't have is the actual risk. The cost model in `config.py` weights
  a bad auto-reply 8x a human touch for exactly this reason.

What I didn't build, on purpose: no web UI, no agent framework, no vector
database, no fine-tuning, no ticketing integration, no ability to actually
take account actions, and no multi-turn autonomous conversation. The agent
reads one message plus whatever prior context is available and produces one
structured decision: intent, reply, route, reason. It never sends
anything. That scope match is deliberate: this is a Twitter-support
proof-of-concept, not a shared-inbox product, and Hiver's actual customers
run longer, private, higher-stakes email threads that this dataset doesn't
represent. I'm not claiming this generalizes to that.

## Golden-set status

`data/golden/golden_v1.jsonl` has all 250 items, all hand-labelled:

1. `src/sampling.py` selects 250 candidate messages from the held-out
   `test_pool` split, stratified across 10 intents with rare/ambiguous cases
   oversampled (see [the codebook](data/golden/codebook.md)).
2. I labelled all 250 by hand in a spreadsheet export
   (`data/golden/golden_labelling.csv`) against the codebook: intent, the
   escalate/auto call, and a one-line reason for each.

Every record carries `label_source: "human"`. The labels, including the
wording of every `escalate_reason`, are mine.

I also rated 80 reply drafts by hand, blind, on a 1–5 rubric across four
dimensions (groundedness, resolution helpfulness, brand voice, safety) in
`data/golden/reply_ratings.jsonl`, `rating_source: "human"`. This is the
independent check on whether the LLM judge (below) can be trusted at all.

## Results vs. two baselines

Three systems evaluated on the same 60 golden items: the **agent**
(retrieval-grounded generation), a **simple baseline** (TF-IDF+logistic
regression for intent, BM25-copy of the closest historical reply for the
draft), and a **trivial baseline** (majority-class intent, one fixed canned
reply, always escalate).

**Intent classification** (accuracy, 95% bootstrap CI):

| system | accuracy | macro F1 |
|---|---|---|
| agent | 63.3% [51.7, 75.0] | 0.643 |
| simple (TF-IDF+LogReg) | 38.3% [26.7, 51.7] | 0.324 |
| trivial (majority class) | 0.0% | 0.000 |

The agent clears the simple baseline by a wide margin, and the simple
baseline clears trivial by a wide margin. The ordering is what you'd hope
for, and neither baseline is a strawman: TF-IDF+LogReg is a real classifier
that a lot of production systems actually run.

**Routing** (precision/recall/F1 against `should_escalate`):

| system | precision | recall | F1 |
|---|---|---|---|
| agent | 0.854 | 0.714 | 0.778 |
| trivial: always escalate | 0.817 | 1.000 | 0.899 |
| trivial: always auto | 0.000 | 0.000 | 0.000 |
| simple baseline (confidence-based routing) | 0.000 | 0.000 | 0.000 |

This table is the one place a reader could get misled if they only look at
the first row. **"Always escalate" beats the agent on F1.** That's not a
quirk of the metric: it's the correct mechanical result of a dataset where
49 of 60 items should escalate. A system that escalates everything is
trivially high-recall, and F1 rewards that here more than it should. The
number that actually matters is what happens on the auto side: of the
19 items the agent routed to auto, 5 were correctly non-escalation-worthy
and 14 should have escalated and didn't, a 71.4% recall on escalation
that's still missing a substantial fraction of real cases (see failure
mode #1 below). "Always escalate" isn't a competitor worth losing to. It
isn't a support agent, it's a null policy. But it's a real reminder that
F1 alone doesn't tell you whether the risky failure mode (auto-answering
something that needed a human) is under control.

**Reply quality** (LLM judge mean scores, 1–5):

| system | groundedness | resolution helpfulness | brand voice | safety |
|---|---|---|---|---|
| agent | 4.87 | 4.57 | 4.87 | 5.00 |
| simple (BM25-copy) | 5.00 | 4.13 | 4.97 | 5.00 |
| trivial (canned) | 5.00 | 3.00 | 5.00 | 5.00 |

Take this table with real caution: see the judge-validation section below.
The judge's own scores here look good for everyone, including the trivial
one-line canned reply, which is itself a signal that the judge isn't
discriminating quality as sharply as its numbers suggest.

## The headline number: coverage at a fixed safety target

The actual question a business cares about: **if you let this system
auto-handle tickets without a human, how many can it safely take, at a
stated error-rate target?** `risk_coverage.py` sweeps a confidence
threshold and reports the highest coverage (fraction auto-handled) that
still meets a 95% safe-auto-reply-rate target.

**Result: coverage = 1.7% (1 of 60 items), threshold = 0.88, at 100%
measured safe-rate.** This is a bad-looking headline number and I'm
reporting it as-is rather than picking a friendlier framing.

Cost-ratio sensitivity (how coverage moves if a bad auto-reply is treated
as more or less costly than a human touch):

| cost ratio (bad auto-reply : human touch) | threshold | coverage | safe-auto-reply rate |
|---|---|---|---|
| 2x | 0.62 | 8.3% | 60% |
| 4x (the configured default) | 0.88 | 1.7% | 100% |
| 8x | 0.88 | 1.7% | 100% |
| 12x | 0.88 | 1.7% | 100% |
| 20x | 0.88 | 1.7% | 100% |

The coverage number is genuinely sensitive between 2x and 4x, then flattens.
The threshold sweep hits a wall around 0.88 where the confidence
distribution just doesn't have many items clustered above it. That's a
feature of this classifier's calibration on 60 items, not a law of nature,
but it does mean "raise the cost ratio further" stops moving the number.

## What is misleading about my headline number?

- **N=60 of 250, not 250.** Groq's free-tier daily token quota (200k TPD)
  doesn't stretch to 250 live generations plus judge calls in one sitting.
  The 60 evaluated are the first 60 items in `golden_v1.jsonl`'s order,
  which is itself a one-time-seeded shuffle from sampling, not cherry-picked.
  Still, 60 items is a small sample, and the coverage number's confidence
  interval on "1 of 60" is wide. I would not present 1.7% as a precise
  estimate of anything; it's a directional result.
- **1.7% coverage is the honest result of a conservative pipeline, not
  proof the agent doesn't work.** The intent and reply-quality numbers
  above show the agent is meaningfully better than both baselines at the
  underlying tasks. The coverage collapse comes from stacking a hard
  guardrail gate, a confidence threshold, and a strict 95% safety target on
  top of that, each individually reasonable, together very conservative.
  A reader who only sees "1.7%" without this context would wrongly conclude
  the agent is broken rather than that the safety bar is set high.
- **The LLM judge that "safe" is partly defined against is itself weakly
  **, see below. The safe-auto-reply-rate in the coverage table
  is computed using judge scores (`score.mean() >= 4.0`), and the judge's
  agreement with my own blind ratings is weak. So "100% safe-auto-reply
  rate" at 1.7% coverage is a real number from a real pipeline, but it
  inherits the judge's own reliability problem.
- **A single annotator, no second rater.** All 250 golden labels and all 80
  reply ratings were done by one person (me). There's no inter-annotator
  agreement number and no independent check on systematic bias in how I
  applied the codebook.
- **The golden set is a stratified challenge sample, not a volume-weighted
  one.** Rare intents and ambiguous cases were deliberately oversampled so
  the system gets stress-tested. None of the accuracy numbers above should
  be read as "this is what happens to a random incoming message" : they're
  a harder test than that.
- **Twitter support isn't Hiver's actual product context.** Public, short,
  low-stakes messages, versus the longer, private, higher-stakes email
  threads Hiver's customers run through a shared inbox. Whatever this system
  proves, it doesn't prove readiness for that environment.

## Judge validation: does the LLM judge agree with a human?

I rated 80 reply drafts blind (`data/golden/reply_ratings.jsonl`) and
compared the LLM judge's scores against mine, alongside two decoy
"judges" built specifically to fail: one that scores purely off reply
length, one that scores randomly. If the real judge doesn't clearly beat
the length-only decoy, that's the honest result to report, not something to
smooth over.

**It doesn't clearly beat the length-only decoy. On three of four
dimensions, the decoy tracks my ratings better than the actual judge does.**

| dimension | real judge (ρ) | length-only decoy (ρ) | random decoy (ρ) |
|---|---|---|---|
| groundedness | 0.226 | 0.531 | -0.019 |
| resolution helpfulness | 0.293 | 0.510 | 0.091 |
| brand voice | 0.265 | 0.598 | -0.017 |
| safety | 0.000 | 0.135 | 0.149 |

(ρ = Spearman rank correlation with my ratings, n=80 for every cell.
Quadratic-weighted kappa tells the same story and is in
`artifacts/judge_agreement_report.json`.)

The safety dimension is the most concerning: the judge's safety scores have
**zero rank correlation** with mine. Looking at the raw scores, the judge
gave almost every reply a 5/5 on safety regardless of content. It isn't
discriminating unsafe replies from safe ones at all in this sample, it's
just defaulting high. That matters because the coverage headline's "safe"
threshold is partly built on judge safety scores.

Separately, I re-judged a 30-item subset with the evidence order swapped in
the prompt (same reply, same evidence, reordered). **26.7% of items flipped
on at least one dimension purely from reordering** — a real reliability
problem independent of the human-agreement numbers.

**Bottom line: this judge, as currently configured, is not reliable enough
to trust on its own.** The reply-quality table above and the "safe" side of
the coverage number should both be read with this in mind. Reporting this
plainly is the point of the exercise: a judge that quietly disagreed with
a human and I didn't check would be worse than an honest weak result.

### Attempted fix: few-shot calibration made it worse

The obvious fix for a judge that's drifting toward generous, uncalibrated
scores is to show it worked examples: real replies, scored by a human,
spanning the range from bad to good. I built this (`src/eval/judge_v2.py`)
using a clean split of the 80 human ratings, 20 held back to build 6
worked examples spanning the score range, 60 never shown to the judge as
an example, used only to re-measure agreement, so the test can't grade on
its own training data. Same evidence-retrieval setup as the original judge,
re-run live and compared on the identical 60 held-out items against the
original zero-shot judge's scores on those same items.

**It made every dimension worse, not better:**

| dimension | zero-shot (v1) | few-shot calibrated (v2) |
|---|---|---|
| groundedness | +0.275 | -0.133 |
| resolution helpfulness | +0.226 | +0.125 |
| brand voice | +0.336 | -0.100 |
| safety | +0.000 | -0.062 |

(Spearman ρ against human ratings, n=60 for both, same holdout set.)

Two dimensions that were weakly positive went negative. My working
hypothesis: `allam-2-7b` is a small model (7B parameters), and stuffing 6
full worked examples plus their scores into the prompt likely overloaded
its ability to reason about the actual new item in front of it, it may be
pattern-matching toward whichever example looks most similar rather than
scoring independently. Few-shot prompting reliably helps larger models;
it isn't guaranteed to help a small one, and here it didn't.

This is a real, reported failure of an attempted fix, not a workaround
that quietly disappeared. The honest conclusion: the fix for this judge
isn't a better prompt on the same small model, it's very likely a larger
or different judge model entirely. See "what I'd do with one more week."
Full numbers: `artifacts/judge_agreement_v2_report.json`.

## Top 5 failure modes

Pulled from the 60-item agent run, real examples, `item_id`s included so
they're traceable back to `artifacts/predictions_agent.jsonl` and
`data/golden/golden_v1.jsonl`.

**1. Missed escalations, the safety-relevant one.** Of 49 items in the
eval set that should escalate, the agent auto-routed 14 of them (29%). Only
6 items were over-escalated in the other direction. Examples: `g_141188`
("nasan na yung #reputation album ha 😒", a missing-content complaint in
Tagalog, auto'd), `g_2865505` ("STOP DELETING MY DAILY MIXES!", an angry
capitalized complaint, auto'd), `g_1593811` (a detailed complaint about a
3,333-track library limit, auto'd). Hypothesis: the escalation call is
being driven mostly by the guardrail's structural triggers (unverifiable
promises, sensitive-data requests) rather than by tone or complaint
severity: an angry customer with a legitimate grievance but no triggering
phrase slips through as "safe to auto-answer" when it probably shouldn't.

**2. Bare/ambiguous messages misclassified as "other" instead of
"not_a_support_request."** 12 of 14 `not_a_support_request` items in the
eval set (86%) were misclassified, almost all short context-free messages:
`g_44312` ("How?"), `g_660571` ("help me"), `g_510521` ("please answer my
DM"). The model treats these as a vague "other" support issue rather than
recognizing there's no actual support content to act on. Hypothesis: the
taxonomy's `not_a_support_request` bucket depends on recognizing *absence*
of a request, which needs more prompt emphasis than a positive category
does. The model defaults to assuming something needs handling.

**3. Questions about availability/timing of something (a launch, a code, a
release) get misread as content-catalog questions.** `g_2898308` ("what's
the presale code for Billie Eilish tomorrow?", gold `how_to_or_usage_question`,
agent predicted `content_availability_or_licensing`), `g_66022` ("I hear
you'll launch in Romania soon? Did I hear right?", same gold label, same
wrong prediction). Both are genuinely asking "is/will X happen," which the
model reads as a content-catalog question because the surface form
resembles "why isn't X available" complaints, which dominate the
`content_availability_or_licensing` training examples. Hypothesis: the
taxonomy conflates "is this available" with "will this happen" — they read
similarly but need different replies (a status check vs. a catalog answer).

**4. Guardrail over-triggers on evidence citation, forcing an unnecessary
escalation.** `g_2943783` (an iOS crash report, no gold escalation need)
got escalated on `INVALID_EVIDENCE_CITATION`: the reply's citation didn't
validate cleanly against retrieved evidence, so the hard gate fired. This
is the safe-direction failure (over-cautious, not under-cautious), but it's
still a real cost: a genuinely simple, safe-to-answer bug report got kicked
to a human because of a citation-formatting mismatch, not because the
content was actually risky.

**5. Reply quality dips specifically on resolution helpfulness, not
safety.** Across the reply-quality table, the agent's resolution
helpfulness (4.57) trails its groundedness and safety (both ~4.9-5.0) by
more than the other systems' internal gaps do. Reading actual drafts during
the blind rating pass, the pattern is generic "please DM us your account
email" replies used even where the customer's message already contains
enough information to give a more specific answer. Example: `g_580943`
("how long does it take to get a response? I need help!") gets a DM-ask
instead of any actual expectation. Hypothesis: the retrieval step is
pulling historical replies that are procedurally safe defaults (ask for
DM) rather than replies that actually resolved a similar issue, because
"safe default" replies are overrepresented in the historical corpus.

## What I'd do with one more week

1. **Replace the judge model before trusting the safety number.** Few-shot
   calibration (20 worked examples from my ratings, tested on a clean
   held-out 60) was the obvious fix, and I built and ran it. It made every
   dimension worse, not better, most likely because `allam-2-7b` is too
   small to benefit from that much in-context calibration. The remaining
   lever is a different, larger judge model, re-validated with the same
   held-out methodology before it's trusted for anything load-bearing.
2. **Run the full 250 golden items, not 60.** Needs either a paid Groq tier
   or spreading the run across several days under the free quota. Would
   tighten every confidence interval in this report and make the coverage
   number trustworthy at more than "directional."
3. **Add a `not_a_support_request` / regional-policy pass to the taxonomy
   prompt.** Failure modes #2 and #3 both look fixable with targeted
   few-shot examples in the classification prompt rather than a redesign.
4. **Get a second labeller on a subset of the golden set**, to get a real
   inter-annotator agreement number instead of reporting "single annotator"
   as an open limitation.
5. **Investigate the escalation-recall gap (#1) specifically.** It's the
   one failure mode with actual safety consequences, and right now I only
   have a hypothesis (guardrail triggers dominating over tone/severity),
   not a fix.
