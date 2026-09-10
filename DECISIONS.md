# Decision log

Non-obvious decisions made while building this, and what each one gave up.
See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full spec these
decisions implement.

1. **Brand chosen by a scripted, pre-declared scoring pass, not by hand.**
   `src/data.py shortlist` ranks brands by outbound volume; a third criterion
   (a regex/length heuristic for "DM us"-style deflection replies) is scored
   before inspection of sample conversations. AmazonHelp had the largest
   outbound volume in the computed shortlist. The shortlist heuristic alone
   does not establish resolution quality. Gave up: choosing solely by volume.

2. **Selected SpotifyCares after reading 30 early-window conversations per
   shortlisted brand, not on the scoring pass alone.** The heuristic score is
   noisy at the message level; an AI assistant inspected examples in
   `reports/brand_review_samples/` before selecting the brand. A complete
   independent human review is not recorded. Gave up: purely automatic selection.

3. **Chronological split (60/20/20 train/dev/test_pool), not random.**
   Temporal ordering limits look-ahead. The current extractor splits direct
   exchanges; full connected-component isolation and corpus-wide near-duplicate
   exclusion still need implementation/verification. Do not claim those
   guarantees from the present labels. Gave up: using later-period examples
   in the training corpus.

4. **Golden-set candidates sampled only from `test_pool`, the latest
   chronological 20%.** Enforces that the retrieval corpus (`train`) is
   temporally prior to everything being evaluated. Gave up: being able to use
   the full dataset for golden-set sampling.

5. **All 250 golden labels are AI-drafted, human-reviewed — disclosed as
   such, not presented as independently hand-written.** A pre-annotator
   model (Qwen, a third family distinct from the generator and judge)
   drafts intent, escalate/auto, and a one-line reason for every candidate;
   Ashraf reviewed all 250 in a spreadsheet and confirmed every one (0/250
   overrides). Every record is tagged `label_source:
   "ai_drafted_human_reviewed"` rather than plain `"human"`, because the
   phrasing of the reason field in particular is the model's, confirmed
   rather than authored — and that's a meaningful difference if asked to
   defend a specific label's wording. Gave up: the cleaner but less accurate
   claim of a fully from-scratch human-written golden set, and the ability
   to measure independent human/suggestion agreement (no held-out blind
   subset exists in the final version) — both traded for review speed on a
   250-item set within the time actually available.

6. **`not_a_support_request` is a mandatory taxonomy bucket, not folded into
   `other`.** TF-IDF/KMeans clustering during taxonomy discovery
   (`src/taxonomy.py`) surfaced multiple near-pure-noise clusters — bare
   mentions, URL-only tweets, "thanks!" follow-ups — that together account
   for roughly 15-20% of messages addressed to the brand handle. Folding
   these into `other` would force-fit noise as a real support intent
   everywhere downstream. Gave up: taxonomy simplicity.

7. **Clustering is a discovery aid; the taxonomy is hand-written.**
   `src/taxonomy.py` prints exemplars from each cluster; the codebook then
   defines intent boundaries and near-misses by hand, not by taking cluster
   IDs as labels directly. Gave up: the speed of accepting cluster output
   as-is.

8. **Deterministic post-hoc guardrails, not prompt instructions, gate
   unsupported commitments.** `src/policy.py` regex-matches for
   refund/compensation/account-action language and sensitive-data requests
   after generation, independent of what the prompt asked for. A reviewer
   can point at the exact rule that fired rather than trusting the model
   followed an instruction. Gave up: some recall on subtler unsupported
   claims a regex can't catch — logged as a known limitation, not hidden.

9. **The escalation confidence threshold is derived from an explicit cost
    model, with a stated sensitivity sweep, not hand-picked.**
    `src/eval/risk_coverage.py` finds the threshold minimising expected cost
    under `COST_BAD_AUTO_REPLY : COST_HUMAN_TOUCH` (default 8:1), and
    separately reports how the headline coverage number moves across a
    2:1-20:1 sweep. Gave up: a single clean threshold number with no caveats
    — replaced with an honest "here's how much this depends on an assumption
    you might disagree with."

10. **The BM25-copy baseline reuses a real historical reply verbatim, chosen
    deliberately as the "simple" baseline rather than a weaker strawman.**
    It's expected to beat the LLM agent on groundedness, since it *is* real
    text a human agent actually sent. A baseline that wins on one dimension
    is more informative than three baselines the system trivially beats.
    Gave up: a cleaner "our system beats every baseline on every metric"
    narrative.

11. **Embeddings fall back to pure scikit-learn (TF-IDF + TruncatedSVD)
    if a neural encoder can't run.** System Python here is 3.14, where torch
    wheels are often unavailable; `sentence-transformers` is preferred but
    not required, and whichever backend actually ran is recorded in
    `artifacts/run_manifest.json`. Gave up: guaranteed use of a neural
    encoder, in exchange for the pipeline actually running on this machine
    without a fragile Python-version dependency.

12. **All LLM calls go through a disk cache keyed on
    `sha256(model + prompt + params)` (`src/llm.py`), and `make reproduce`
    is designed to run entirely from that cache with no network or API
    keys.** This is what makes the 15-minute reproduction claim credible
    rather than aspirational — a reviewer with no Kaggle account and no API
    keys can still regenerate every number. Gave up: some code simplicity,
    in exchange for a reviewer being able to verify results are not
    hand-edited.

13. **No agent framework, vector database, or orchestration layer.**
    Retrieval is a ~60-line BM25 index; the agent is one structured LLM call
    plus deterministic guardrails. The brief states candidates will be asked
    to explain and modify their own code live — every added dependency is
    something to defend in that conversation, and none of LangChain /
    LlamaIndex / a vector DB service solves a problem this dataset actually
    has at this scale. Gave up: looking more "sophisticated" on paper.
