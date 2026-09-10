# Hiver take-home implementation plan

**Labeling status (2026-09-10):** The golden set is `data/golden/golden_v1.jsonl`:
150 items reviewed and confirmed by hand via `src/golden_csv.py` (see
DECISIONS.md #5 for exactly how), 50 remaining items to be labelled
independently via `src/label_tui.py --blind-only`.

Status: core pipeline, retrieval, agent, guardrails, baselines, taxonomy,
and 150/200 golden labels are built. Optimize for evidence, a small runnable
system, and code Ashraf can explain and modify live. No plan can guarantee a
shortlist or offer.

## 1. Objective and boundaries

Build a support-agent prototype for one brand in Customer Support on Twitter. For each inbound message and its available prior context, return an intent, an evidence-grounded reply draft, and `auto` or `escalate` with a concrete reason. Never actually send replies.

“Auto” means a draft is suitable for a public response without account access under our written capability policy. It does not mean the customer’s issue was resolved. Historical brand replies are examples of support behavior, not verified resolutions or current policy. Evaluate within the historical dataset setting; do not claim deployment readiness for today's customers.

The differentiator is a traceable evaluation: a reviewer can follow a result to a prediction, evidence, human label, and original tweet ID; rerun the arithmetic; and see where the proposed automation fails. A modest honest result is better than an unsupported 95% claim.

| Assignment requirement | Concrete artifact |
|---|---|
| One brand; intent, draft, routing and reason | CLI and typed prediction schema |
| Runnable pipeline; headline reproducible in <15 minutes | README, locked dependencies, `make reproduce`, separate live path |
| 150–250 hand-labelled examples; sampling note | 200 final test examples, codebook and annotation records |
| Automated metrics and LLM judge with human agreement | Evaluation harness, rubric, blinded human ratings and agreement report |
| Framing, two baselines, five real failures, misleading headline, next week | Concise `REPORT.md`, target six pages |
| 10–15 non-obvious decisions | `DECISIONS.md`, 12 decisions with tradeoffs |
| Borrowed work cited; submission through form | Sources, attribution, repo/report links submitted by Ashraf |

No web UI, agent framework, vector database, fine-tuning, ticketing integration, account actions, secondary dataset, or autonomous multi-turn workflow. Prior conversation context is supported. CLI inspection and saved example traces are sufficient.

## 2. Establish prerequisites before implementation

The previous plan claimed credentials and Python availability were verified. Treat those claims as unverified until a local preflight records the actual environment; never print secret values.

1. Confirm deadline, available human annotation time, and API budget early; do independent work while awaiting these constraints. Default planning budget: seven working days, with 8–12 hours reserved for human annotation/review across sessions. Time a pilot before making promises.
2. Use Python 3.12 and a lockfile. Keep dependencies to numpy, scikit-learn, a schema validator, plotting, pytest and provider clients. BM25 can use a small maintained package with attribution. No torch dependency in the required path.
3. Confirm dataset access and inspect its schema, provenance and redistribution terms before committing excerpts. Do not assume public availability grants arbitrary redistribution. Record the applicable source terms and date; if necessary provide permitted excerpts/IDs through the allowed distribution route while retaining a runnable review bundle.
4. Run one small structured-output call per provider, verifying account access, model IDs, rate limits and costs. The generator candidate is Groq `openai/gpt-oss-120b`; select an available Gemini judge after preflight. Pin actual IDs, request parameters and provider metadata in the manifest. Model choice is settled on development data before testing.
5. Prefer a different generator/judge family as a precaution. This does not eliminate shared biases. No third model is required: final golden labels are human-first, without model suggestions.

Official model catalogs were consulted during plan review; catalog presence is not evidence of account access. The supplied submission-form URL could not be read through the web tool, so the pasted assignment remains the authoritative brief. Check the form for additional fields/deadline before submission; no extra judging criteria are assumed.

## 3. Data, brand selection and leakage controls

### Select a viable brand without evaluating the agent

Download the raw dataset once during development; keep it gitignored. Stream the CSV in chunks. Use explicit reply links and brand authorship to associate customer messages; `inbound=true` alone does not identify a brand, and mentions alone can be ambiguous.

Shortlist the five largest eligible brand accounts by linked-message count. Inspect an early-window sample of 30 conversations per candidate, before opening the test window. Record counts and examples of:

- usable customer context and missing parents;
- actionable general guidance versus private-channel handoffs;
- diversity of support intents and account-specific issues;
- observable customer confirmation, if any, separately from a brand reply.

Choose the brand with the most reusable public guidance among candidates with enough data for all splits and at least several distinct intents. Break ties by usable volume. Write the criteria before inspection and the actual selection rationale afterward. Do not assume AppleSupport is largest, that a particular brand mostly deflects, or that a thread ending with a brand message proves resolution. Limit this step to a few hours.

### Reconstruct safely

Use `in_response_to_tweet_id` edges for the graph; use `response_tweet_id` as a consistency check. Detect duplicate IDs, cycles, missing parents, branching, cross-brand conversations and inconsistent timestamps. Keep explicit eligibility flags and an exclusion-count ledger. Quarantine ambiguous cross-brand records instead of guessing.

An evaluation unit is one customer message with ancestor context available at its timestamp. Use at most one seeded, eligible inbound message per connected conversation component for the labeled samples; the sampling population is therefore conversation episodes, not all tweet volume. Document that restriction. Include eligible unanswered customer messages: requiring a brand reply would select only messages the brand handled. Historical target replies are optional metadata, never agent input.

Each record contains message ID, component ID, timestamp, brand, customer text, ancestor messages with author roles and timestamps, and separate historical-reply metadata. Context follows the ancestor path, never siblings or future turns. Retrieval records can contain historical customer/brand pairs, but only from the training period.

Scrub emails, phone numbers, order/account identifiers and customer handles. Preserve meaningful product names, versions and amounts where appropriate; replacing all numbers destroys issue distinctions. Convert URLs into domain/path-category metadata for evidence inspection and omit personal query strings. A historical URL is not automatically safe or current enough to put into a new draft. Retain source IDs for traceability; do not commit raw secrets or obvious personal details. Document residual redaction limitations.

### Freeze three chronological partitions

Build components before splitting. Choose timestamp boundaries from data counts without model evaluation: approximately 60% historical training/retrieval, 20% development, 20% final test pool. Quarantine components that cross a boundary. All training retrieval pairs must predate the development boundary, and the fixed index remains unchanged for final testing.

Normalize and fingerprint customer text plus context, then find near duplicates using token shingles and a declared similarity cutoff. For the modest selected-brand sample, use indexed candidate matches rather than an all-pairs comparison of the full dataset. Detect duplicates across partitions and remove later duplicates from eligible dev/test pools; do not move future records into training. Do not merge conversations merely because their generic brand reply is identical. Report removals and disclose that this evaluates novel episodes rather than repeated traffic.

Cap the retrieval corpus at 5,000 eligible training pairs, selected with a fixed seed independently of test outcomes; keep their required ancestor context. Record actual counts and the sampling rule. Increase to at most 10,000 only if the development retrieval audit shows a material benefit and local runtime remains acceptable. This bounds BM25 memory, artifact size and preprocessing; the full raw dataset is never part of reviewer setup.

Freeze manifests containing IDs, hashes, seed, eligibility rules, timestamps and duplicate decisions. Assert disjoint component IDs and no future context. Never put test answers, target replies or final gold labels into prompts, taxonomy discovery, the retrieval index or baseline training.

## 4. Human labels and taxonomy first

Read 80–100 training examples and define roughly 6–8 substantive intents, plus `other_or_unclear` and `not_support`. Use concrete observed categories; do not commit to generic billing/login labels before reading the chosen brand. Clustering is optional only if manual inspection leaves a real ambiguity.

The codebook defines each intent, positive examples, a near miss, multi-intent precedence, missing-context handling and the routing policy. Rants that contain a support request remain support requests. Record a secondary intent when useful; evaluate the primary intent consistently. Taxonomy changes occur during the pilot, before final test labeling.

### Explicit annotation budget

| Set | Count | Purpose |
|---|---:|---|
| Training | 150 human-first labels | Fit simple classifier; taxonomy examples are allowed here |
| Development | 60 human-first labels | Prompts, retrieval choices, routing thresholds, rubric pilot |
| Final representative test | 150 human-first labels | Seeded uniform sample of eligible final-window episodes |
| Final challenge test | 50 human-first labels | Rare/ambiguous, context-dependent, account/private-data and multi-intent cases |

The two final sets together are the required **200-example golden set**. Keep development and training labels in separate files. Challenge cases are sampled from the remaining test pool using declared text/context rules, then human-labeled; never use model failures to choose the original challenge set. Record strata, candidate counts and exclusions. Report the 150 representative cases and 50 challenge cases separately; do not pool them into a volume headline. Unknown final intents map to `other_or_unclear` rather than causing post-test taxonomy revision.

Ashraf labels every item without seeing model suggestions, candidate system outputs or future brand replies. A plain terminal form or CSV workflow is enough. Record labeler, timestamp, codebook version, rationale and uncertain cases; no keystroke surveillance is needed. A coding assistant may prepare the form and validate completeness but must not fabricate human labels or ratings.

Labels include `intent`, optional `secondary_intent`, `must_escalate`, reason codes, ambiguity/context flags, and short notes on acceptable guidance or prohibited claims. `must_escalate=false` means an adequate reply could be sent within policy; it does not certify any generated reply.

Blindly repeat 20–30 labels in a later session to describe intra-rater consistency. If a second person is available, independently label 30 and record disagreements/adjudication, but do not make that availability a blocker. Neither repeat labeling nor a second rater is replaced by another LLM. Single-author ground truth is an explicit limitation.

## 5. Small system and strong baselines

### Shared capability policy

Auto-handling is restricted to public, general guidance supported by applicable historical evidence. Account access, identity verification, refunds/compensation, order-specific actions, policy exceptions, security concerns, sensitive personal information and materially missing context require escalation. No claim that an action was performed. The existence of a refund in another customer's history never authorizes a refund for this customer.

Even escalated cases receive a short safe acknowledgement and a reason describing what a human must check. A private-channel handoff is not counted as automated resolution. Route `not_support` conservatively to escalation/review within the assignment's two-route interface; disclose the resulting coverage cost.

### Baselines

1. **Trivial:** majority training intent; fixed safe acknowledgement; always escalate. Also report an explicitly unsafe auto-all routing reference to expose the tradeoff, not as a recommended policy.
2. **Simple:** word/character TF-IDF logistic-regression intent classifier trained on the 150 human labels; BM25 top-one historical reply, scrubbed of customer identifiers; classifier-confidence threshold for routing plus the same capability checks. Missing evidence falls back to escalation. A copied reply can be irrelevant or promise another customer's outcome, so it is not automatically grounded or safe.
3. **Proposed:** BM25 top-five retrieval and one structured generator call with taxonomy, customer context and cited evidence. Use the same frozen evidence corpus and shared deterministic checks.

Start with BM25, not hybrid retrieval. Preserve private-channel examples as escalation evidence rather than deleting them. Tag evidence as general guidance, handoff or account-specific action. Generation instructions distinguish these uses.

One optional ablation: the same generator without retrieved evidence, using the same input/policy and fixed prompt variant. It isolates the contribution of historical grounding. Add dense retrieval only if a development retrieval audit shows lexical mismatch is a major problem, and record the change before final evaluation. No general embedder framework is required.

### Typed result

```json
{
  "message_id": "example-id",
  "intent": "chosen_taxonomy_value",
  "reply_draft": "...",
  "evidence": [{"source_message_id": "historical-id", "supporting_span": "..."}],
  "route": "escalate",
  "reason_codes": ["ACCOUNT_ACCESS_REQUIRED"],
  "route_reason": "A human must inspect the account before confirming the charge.",
  "ranking_score": 0.64
}
```

Validate schema/enums, source membership, exact supporting spans and length bounds. Span existence verifies citation integrity, not semantic support. Humans and the judge check applicability and entailment. Input/retrieved text is untrusted data, never instructions. Do not execute links or tools from it.

Fail closed on invalid output, timeout, unknown citations, missing evidence, private-data leakage, or detected unauthorized promises. Guardrails can miss paraphrases; report their measured failures rather than calling them a safety proof. Keep final routing in code; the model cannot override a hard veto.

Use the shared TF-IDF classifier’s maximum probability as the initial ranking score for both tunable systems; label it an uncalibrated ranking signal. For the proposed system, a disagreement between the classifier and generated primary intent forces escalation. This score measures intent certainty, not reply correctness; evidence and capability checks remain separate. Inspect BM25 lexical overlap and evidence relevance on development examples. BM25/RRF ranks and margins are not probabilities; an RRF top score cannot establish absolute relevance. Do not fit a multi-feature safety classifier on a tiny unspecified development set.

## 6. Freeze the operating policy before final evaluation

### Development usefulness gate

Before spending the full annotation/evaluation budget, inspect 20 development episodes and obtain a working trace for each. Confirm at least some genuinely useful public guidance is possible within the chosen policy, that retrieved evidence is applicable, and that escalation reasons identify the actual missing capability. A system that escalates everything may be safe, but has not demonstrated useful automation.

If the development operating policy selects zero coverage, inspect the cause before freezing: data lacks public guidance, retrieval misses it, the intent classifier is weak, or the policy blocks legitimate general advice. Permit one focused development iteration on the dominant cause; never relax account-action restrictions to improve the number. If zero coverage persists, report an assisted-drafting result and explicitly state that auto-handling was not demonstrated. Treat this as a competitive weakness, not a success hidden behind undefined safety.

On the same 20 examples, compare proposed and simple replies for applicable advice, useful specificity and unsafe claims. This is a development diagnostic, not final evidence of superiority. If generation adds no value, improve the demonstrated failure once rather than adding infrastructure. Preserve the final paired comparison even if it favors the baseline.


On the 60 development episodes, human-rate drafts from both tunable systems (simple and proposed) for safe acceptability after a rubric pilot; budget up to 120 development reply ratings. Reuse identical draft ratings where applicable. Use a predeclared small confidence-threshold grid, e.g. `{0.5, 0.6, 0.7, 0.8, 0.9, 1.01}` where 1.01 escalates all. Shared hard vetoes apply at every threshold. Tune each nontrivial system on its own development outputs using the same objective.

Primary selection rule: choose maximum coverage among thresholds with observed development unsafe-auto rate ≤5% and at least 20 auto cases. Ties choose the higher threshold. If none qualifies, choose escalate-all. This is a development heuristic, **not a certified 95% safety guarantee**. Record dev counts and intervals; small denominators may make the target unprovable.

Freeze corpus, taxonomy, prompts, code, thresholds, models, rubric and test IDs before generating or inspecting final outputs. Record a manifest/hash. Development model or judge failures are fixed here. Final failures are reported; a post-test fix is a separately versioned exploratory result, not a fresh untouched test.

Cost analysis is secondary: report `8 * unsafe_auto_count + escalation_count`, divided by N, with ratios 2, 5, 8 and 20 as illustrative assumptions. Do not describe them as measured business costs. Keep the selected policy fixed for this sensitivity table. A cost-optimal threshold does not imply a 95% safe rate.

## 7. Evaluation and human validation of the judge

### Rubric

Give the judge the incoming message, allowed prior context, shared capability policy, candidate reply, claimed evidence and a fixed common evidence packet from the frozen retriever. Never show the held-out historical answer as required wording or gold truth. Blind system names, routing confidence and human labels.

Score 1–5 with explicit anchors at **every score**, especially the pass boundary at 4:

- Evidence support/applicability: factual advice is supported and matches the situation.
- Helpfulness: useful next step that addresses the actual request; clarification/handoff can be appropriate.
- Safety: no unauthorized action, sensitive-data request, unsupported commitment or dangerous advice.
- Brand tone: appropriate tone based on training examples, reported separately from safety.

Add binary `acceptable_for_auto`, a critical-violation code and short rationale. Require support, helpfulness and safety ≥4, no critical violation and policy eligibility for auto acceptability; brand tone does not define safety. Supply sample anchor cases from development and version the rubric. Judge abstentions/invalid outputs are reported as missing and conservatively fail acceptance; never drop them from denominators.

### Human judgment that matters

1. Pilot the rubric on approximately 20 development replies; resolve confusing definitions before freezing it.
2. Human-rate **every proposed-system auto draft in the representative test**. This directly measures the deployed policy's most important risk; auditing only ten easy examples is insufficient. Hide model/judge identity and scores, and interleave these with the agreement items.
3. Independently human-rate all three required systems' drafts on the same 30 randomly selected representative messages: **90 reply ratings**, including escalations. Keep the paired structure for comparisons. These can overlap the auto audit to avoid duplicate work.
4. After freezing ratings, compare the judge with the human: per-dimension weighted Cohen's kappa and raw agreement; binary acceptability confusion matrix, especially false acceptance `human reject / judge accept`, with explicit counts and confidence intervals. Spearman correlation is optional and cannot substitute for dangerous-error counts. Undefined kappa under constant scores is reported as undefined.
5. Bootstrap agreement/comparative intervals by message to preserve dependence between systems. Report that the agreement sample is small and has one human rater. Different model families do not validate a judge; measured agreement does.

Single-reply scoring has no response-order position swap to test. Optional repeatability checks can re-score 10 replies, but length-decoy experiments and artificial random judges are lower priority than human auditing. If judge false acceptance is substantial, report automated quality as exploratory and base safety conclusions on the human audit. Judge–human agreement is evidence about measurement reliability, not a mathematical ceiling on system performance.

### Metrics and exact claims

For each system, report on representative and challenge sets separately:

- Intent accuracy, macro-F1, per-class precision/recall/F1 and support. Use fixed label order; state handling of zero-support classes. Routing confusion matrix and must-escalate recall.
- Auto coverage `auto_count / all_messages`; unsafe-auto rate among autos; missed mandatory escalations among mandatory escalations. Label each denominator.
- Judge dimension distributions and automatic pass rates; human-audited proposed-policy acceptability on representative autos; paired human quality differences on the 30-message subset.
- End-to-end correct-auto rate: correct intent AND appropriate routing AND acceptable reply, as a separate metric. A safe draft with a wrong intent is not full task success.
- Latency median/p95, tokens, estimated API cost and failed-call counts from actual uncached runs; specify sample size and environment.

Use seeded 2,000-resample bootstrap intervals for macro-F1 and paired differences, and Wilson or exact binomial intervals for rates. For zero unsafe events among m autos, report a nonzero one-sided 95% upper bound `1 - 0.05 ** (1/m)`. With 20 flawless autos that bound is about 14%; even 59 flawless autos only just bring it below 5%. At zero autos, safety precision is undefined, never 100%.

Headline template: **“On 150 held-out eligible conversations, the frozen policy auto-drafted A/150 (C%). Human review found B/A unacceptable (95% interval …).”** Add the sampling restriction in the adjacent sentence. Do not extrapolate this to all inbound volume, resolution rate, time saved or current production safety.

Plot risk versus coverage, with the preselected operating point marked and counts/uncertainty. Other test thresholds are exploratory, judge-estimated diagnostics; they cannot be used to choose the headline threshold. Human validation at the operating point does not validate every point of the curve. If a 95% safe claim is unsupported, say so even when the point estimate is 100%.

Audit retrieval separately on 20 development messages: whether top-five includes applicable guidance and why retrieval fails. On final failures distinguish retrieval miss, unsupported generation, incorrect intent, routing error and evaluation disagreement rather than assuming every bad reply is an LLM problem.

## 8. Artifacts and reproduction

Keep modules small and concrete:

```text
README.md, REPORT.md, DECISIONS.md, IMPLEMENTATION_PLAN.md
pyproject.toml, lockfile, Makefile
src/data.py          # extraction, graph, redaction, partitions
src/label.py         # simple annotation workflow and validation
src/retrieve.py      # BM25 and evidence records
src/baselines.py
src/agent.py         # prompt, typed output, deterministic policy, CLI
src/llm.py           # bounded retries, cache, usage logs
src/evaluate.py      # metrics, judge agreement, figures
prompts/             # versioned generator and judge instructions
config/              # taxonomy, capability policy, frozen thresholds
data/                # permitted scrubbed sample, manifests, train/dev/gold labels
artifacts/           # predictions, evidence IDs, judge outputs, human ratings, results
reports/figures/
tests/
```

Cache keys cover exact request content, model/provider, parameters and prompt/schema version. Manifests include source and input hashes, split IDs, dependency versions, code revision/content hash, seed, timestamps, failed requests and usage. Raw provider responses are retained after redaction where permitted. Hashes detect mismatches; they do not prove results were never edited. Never claim live/cache agreement proves authenticity or demand exact equality from nondeterministic APIs.

- **`make reproduce`:** recompute metrics and figures from committed frozen predictions/labels; verify hashes, counts, citation membership and results against a machine-readable expected-results file. Rerun the simple baseline locally as an additional check. Fail on missing artifacts or unexpected numerical differences. No API keys/downloads during this command.
- **Setup:** README gives a clean Python 3.12 environment installation command. Dependency installation normally requires network; do not claim a clean clone works offline unless wheels are bundled. Time **setup plus reproduction** on a clean environment, target <15 minutes; aim for metric replay <3 minutes.
- **`make live-smoke`:** fresh, cache-bypassed inference on five fixed examples with keys and usage limits. Exposes actual runnable agent behavior.
- **`make live`:** regenerate all selected-system predictions/judgments into a new run directory using explicit limits; never overwrite frozen results. Full live regeneration may exceed 15 minutes and is clearly distinguished from metric replay.
- **`make demo`:** select a saved example for an offline trace, or accept new text in explicit live mode. Print intent, evidence, draft and route/reason. Choose actual brand examples after brand selection.
- **`make test`:** focused tests for graph branches/missing parents/cycles, chronological context and split isolation, redaction, unknown citations, invalid schema/timeouts, unauthorized commitments despite historical precedent, no-evidence escalation, metric arithmetic/zero denominators, threshold freezing and stale-cache rejection. Include an end-to-end fixture with a stub provider.

The README opens with one results table and three linked traces: useful auto guidance, a necessary escalation, and an actual failure. Each trace exposes the message, evidence, output and human judgment. A reviewer should find the claim, its limitation and the reproduction command within a minute. Select illustrative traces after evaluation and label them illustrative, not representative.

The README explicitly calls replay “recomputing metrics from recorded model outputs”; it does not imply offline LLM inference. Report measured runtime, hardware, install conditions, package versions and whether network was used. No unnecessary full-dataset requirement for reviewers.

## 9. Report and decision log

Target a six-page-equivalent report with:

1. Brand, observed data, policy/capability boundaries, definition of good and omitted scope.
2. Sampling, taxonomy, temporal split, human labeling and judge validation, including disagreements.
3. Results table against both baselines, operating-point counts/intervals, risk–coverage plot and small cost table. If the simple baseline wins, explain what the result implies rather than hiding it.
4. Five observed failure modes: redacted real example ID/text, prediction, evidence, expected behavior, consequence, frequency and a testable hypothesis. Tag all evaluated failures under a declared primary-mode scheme; if there are fewer than five distinct observed modes, say so and show limitations without inventing cases. Synthetic robustness examples are separate from dataset performance.
5. Exact required heading: **“What is misleading about my headline number?”** Cover selected brand/eligibility, removed duplicates, historical data and unknown private outcomes, one human rater, sample uncertainty, judge errors, development tuning and replay versus fresh inference. Twitter being public or short does not make it inherently low-stakes.
6. One more week: prioritize a second annotator, a larger untouched temporal set for safety precision, targeted retrieval corrections and current policy validation. Choose from actual failures.

Write 12 decision-log entries with decision, evidence/reason and cost: brand; evaluation unit; temporal/duplicate exclusions; taxonomy; human-first labels; separate challenge set; BM25; capability limits; copied-reply baseline; threshold freeze; judge audit; replay/live separation. Record actual decisions, not retrospective claims that every choice was inevitable.

Cite source dataset, reused libraries/models/prompts and methods. Disclose AI assistance accurately. Avoid claims about what other candidates will do or what guarantees employment.

## 10. Execution order and stop rules

| Stage | Work and exit evidence |
|---|---|
| Day 1 | Preflight, source terms, brand sample, graph fixtures, freeze partitions; manual taxonomy pilot |
| Day 2 | Training labels and codebook, simple baseline, end-to-end generator trace |
| Day 3 | Dev labels, retrieval audit, draft/rubric pilot, routing policy; final human labeling begins |
| Day 4 | Finish 200 gold labels, freeze all configurations; run untouched test once |
| Day 5 | Human auto audit, 90 paired reply ratings, judge agreement, metrics and actual failure analysis |
| Day 6 | Report, 12 decisions, attribution, clean-environment reproduction and tests |
| Day 7 | Adversarial review, rehearsal and submission buffer; post-test changes labeled exploratory |

The schedule is provisional until a timed annotation pilot; 410 message labels, up to 120 development reply ratings, 90 paired test ratings and up to 150 proposed auto audits (with overlap) are real work. Time 10 message labels and 10 reply ratings, extrapolate the remaining workload, and reserve a 25% buffer. Increase the initial 8–12-hour human budget if the pilot requires it. If time is tight, remove optional ablations, dense retrieval, extra plots and extra judge diagnostics first. Never replace human labeling with synthetic labels, silently use test data for tuning, or drop a mandatory deliverable to polish a demo. If the required human work cannot fit, surface that constraint immediately.

Before submission Ashraf should explain without notes: graph/context leakage; why historical replies are not resolution truth; the baseline's training labels; routing versus reply acceptability; the unsafe-rate denominator and interval; judge false acceptance; and what replay proves. Rehearse a small code change such as a new escalation rule with a meaningful test.

## 11. Acceptance checklist

- [ ] One actual brand selected with data evidence; no unsupported brand assertions.
- [ ] All three required outputs and safe failure behavior work on fresh CLI input.
- [ ] 200 real human-labeled final examples, separate from 150 train/60 dev, with sampling/codebook records.
- [ ] Frozen partitions, inputs, policy and thresholds; no future replies or target labels in agent inputs.
- [ ] Both baselines evaluated fairly; results use explicit denominators and uncertainty.
- [ ] Human-reviewed proposed autos and ≥90 paired reply ratings (overlap allowed); judge false acceptance and per-dimension agreement reported.
- [ ] Primary headline reports observed frozen-policy performance, not test-selected 95% safety or claimed resolution.
- [ ] Focused tests and clean-environment setup/replay pass within the assignment's 15-minute limit; fresh live smoke succeeds.
- [ ] Required report sections, five evidence-based failures where observed, 10–15 decisions and attribution are complete.
- [ ] Ashraf can inspect an individual result and defend/modify the code live.
- [ ] Repo/report access and form requirements checked; submit through the supplied form, not email.

## Sources for implementers

- Assignment: the user-supplied Hiver brief; [submission form](https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105).
- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter): inspect downloaded metadata and source terms during preflight; web preview did not expose the dataset card during review.
- [Groq supported models](https://console.groq.com/docs/models) and [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models): verify current availability and pin actual run IDs.
- [scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html): consult during implementation for metric definitions and averaging behavior.

This plan is ready to implement subject to preflight and actual annotation capacity. The system's readiness must be judged from its resulting evidence, not from the plan's ambition.
