# Decision log

Non-obvious decisions made while building this, and what each one gave up.
See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full spec these
decisions implement.

1. **Brand chosen by a scripted, pre-declared scoring pass, not by hand.**
   `src/data.py shortlist` ranks brands by outbound volume; a third criterion
   (a regex/length heuristic for "DM us"-style deflection replies) is scored
   *before* any manual review. AppleSupport, the largest brand in the
   dataset, was never a serious candidate for this reason — its replies are
   overwhelmingly deflection to a private channel, which would starve the
   grounded-drafting task of real resolution content. Gave up: a brand with
   more raw volume than SpotifyCares.

2. **Selected SpotifyCares after reading 30 early-window conversations per
   shortlisted brand, not on the scoring pass alone.** The heuristic score is
   noisy at the message level; a human read the actual samples in
   `reports/brand_review_samples/` before committing. Gave up: a fully
   automated selection with no human-in-the-loop check.

3. **Chronological split (60/20/20 train/dev/test_pool), not random.**
   Random splitting would leak near-duplicate, contemporaneous tweets into
   the retrieval corpus and inflate every groundedness number — the retriever
   would find a near-copy of the exact eval item. Split at the connected-
   component level so no conversation straddles a boundary. Gave up: a larger
   effective training corpus, since the historical corpus can't include
   anything from the eval window.

4. **Golden-set candidates sampled only from `test_pool`, the latest
   chronological third.** Enforces that the retrieval corpus (`train`) is
   temporally prior to everything being evaluated. Gave up: being able to use
   the full dataset for golden-set sampling.

5. **Three different model families for generator / judge / pre-annotator**
   (Groq gpt-oss / Gemini 2.5 Pro / Groq Qwen). If the pre-annotator shared a
   family with the generator, golden labels would be biased toward the
   system under evaluation. If the judge shared a family with the generator,
   self-preference bias becomes an unanswerable objection. Gave up: some
   convenience — Groq and Gemini have different SDKs, cache formats, and
   rate limits to manage.

6. **Golden-set labelling is hybrid, not pure-human or pure-model, and the
   split is disclosed.** 150 of 200 items show a pre-annotator suggestion
   the human accepts or overrides; 50 are labelled blind with no suggestion.
   Comparing human-vs-suggestion agreement on the suggested half against the
   blind half gives a measured anchoring-bias number instead of an assumed
   one. Gave up: a cleaner "100% independently human-labelled" claim, in
   exchange for being able to quantify how much the process itself biased
   the ground truth.

7. **`not_a_support_request` is a mandatory taxonomy bucket, not folded into
   `other`.** TF-IDF/KMeans clustering during taxonomy discovery
   (`src/taxonomy.py`) surfaced multiple near-pure-noise clusters — bare
   mentions, URL-only tweets, "thanks!" follow-ups — that together account
   for roughly 15-20% of messages addressed to the brand handle. Folding
   these into `other` would force-fit noise as a real support intent
   everywhere downstream. Gave up: taxonomy simplicity.

8. **Clustering is a discovery aid; the taxonomy itself is hand-written.**
   `src/taxonomy.py` prints cluster exemplars; a human then wrote
   `data/golden/codebook.md` with definitions and a near-miss for each
   intent. "I ran k-means and used the clusters as my labels" is a weak
   answer to a live code-review question. Gave up: the speed of taking
   cluster IDs as the taxonomy directly.

9. **Deterministic post-hoc guardrails, not prompt instructions, gate
   unsupported commitments.** `src/policy.py` regex-matches for
   refund/compensation/account-action language and sensitive-data requests
   after generation, independent of what the prompt asked for. A reviewer
   can point at the exact rule that fired rather than trusting the model
   followed an instruction. Gave up: some recall on subtler unsupported
   claims a regex can't catch — logged as a known limitation, not hidden.

10. **The escalation confidence threshold is derived from an explicit cost
    model, with a stated sensitivity sweep, not hand-picked.**
    `src/eval/risk_coverage.py` finds the threshold minimising expected cost
    under `COST_BAD_AUTO_REPLY : COST_HUMAN_TOUCH` (default 8:1), and
    separately reports how the headline coverage number moves across a
    2:1-20:1 sweep. Gave up: a single clean threshold number with no caveats
    — replaced with an honest "here's how much this depends on an assumption
    you might disagree with."

11. **The BM25-copy baseline reuses a real historical reply verbatim, chosen
    deliberately as the "simple" baseline rather than a weaker strawman.**
    It's expected to beat the LLM agent on groundedness, since it *is* real
    text a human agent actually sent. A baseline that wins on one dimension
    is more informative than three baselines the system trivially beats.
    Gave up: a cleaner "our system beats every baseline on every metric"
    narrative.

12. **The TF-IDF intent classifier (used both as a baseline and as one
    confidence signal for routing) is fit on pre-annotator-labelled
    train/dev data, never on the golden set.** Fitting a confidence combiner
    on the same data it's evaluated against is the most likely way to
    accidentally inflate the headline number. Gave up: a slightly stronger
    baseline classifier, in exchange for a defensible train/eval boundary.

13. **Embeddings fall back to pure scikit-learn (TF-IDF + TruncatedSVD)
    if a neural encoder can't run.** System Python here is 3.14, where torch
    wheels are often unavailable; `sentence-transformers` is preferred but
    not required, and whichever backend actually ran is recorded in
    `artifacts/run_manifest.json`. Gave up: guaranteed use of a neural
    encoder, in exchange for the pipeline actually running on this machine
    without a fragile Python-version dependency.

14. **All LLM calls go through a disk cache keyed on
    `sha256(model + prompt + params)` (`src/llm.py`), and `make reproduce`
    is designed to run entirely from that cache with no network or API
    keys.** This is what makes the 15-minute reproduction claim credible
    rather than aspirational — a reviewer with no Kaggle account and no API
    keys can still regenerate every number. Gave up: some code simplicity,
    in exchange for a reviewer being able to verify results are not
    hand-edited.

15. **No agent framework, vector database, or orchestration layer.**
    Retrieval is a ~60-line BM25 index; the agent is one structured LLM call
    plus deterministic guardrails. The brief states candidates will be asked
    to explain and modify their own code live — every added dependency is
    something to defend in that conversation, and none of LangChain /
    LlamaIndex / a vector DB service solves a problem this dataset actually
    has at this scale. Gave up: looking more "sophisticated" on paper.
