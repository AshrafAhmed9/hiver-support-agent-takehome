# Decision log

Historical implementation notes, with labeling provenance corrected below.
Some entries describe intended architecture rather than completed verification;
the current limitations in `data/labels/README.md` take precedence for evaluation.
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

5. **Preserve per-item model provenance when quotas force a model change.**
   Initial Qwen labels are archived separately. The main reference-label run
   uses Gemini 2.5 Flash, Gemini 3 Flash Preview and Qwen 3.8 27B, with counts in
   the manifest. The Gemini portion shares a family with the configured judge. Gave up: a uniform
   annotator and the originally intended three-family separation. Different
   families would not by themselves prove unbiased evaluation anyway.

6. **Reference labels are AI-assigned with explicit provenance.** The user
   authorized model labeling after discussing disclosure. `src/ai_label.py`
   writes model/provider, rationale, uncertainty and `human_reviewed: false`
   on each train/dev/evaluation label. Old suggestions are not shown to this
   annotator. No human label session or measured anchoring effect occurred.
   Gave up: independent human ground truth and any judge–human agreement claim.

7. **`not_a_support_request` is a mandatory taxonomy bucket, not folded into
   `other`.** TF-IDF/KMeans clustering during taxonomy discovery
   (`src/taxonomy.py`) surfaced multiple near-pure-noise clusters — bare
   mentions, URL-only tweets, "thanks!" follow-ups — that together account
   for roughly 15-20% of messages addressed to the brand handle. Folding
   these into `other` would force-fit noise as a real support intent
   everywhere downstream. Gave up: taxonomy simplicity.

8. **Clustering is a discovery aid; the taxonomy is AI-authored.**
   `src/taxonomy.py` prints training exemplars. The working codebook defines
   intent boundaries and near-misses separately from cluster IDs. Independent
   human validation is not recorded. Gave up: accepting cluster IDs directly.

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

12. **Separate AI training, development and evaluation label files.**
    The 150 training labels are available for classifier fitting; development
    labels are for development diagnostics, and challenge labels for final
    comparison. Label generation does not fit the classifier or tune it on
    the challenge set. Gave up: using all 410 labels for training.

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

16. **The 150 non-blind golden labels were bulk-accepted, not confirmed
    one-by-one in the TUI.** Ashraf reviewed all 150 pre-annotator
    suggestions in a CSV export and reported full agreement; rather than
    re-clicking through each item in the terminal tool, `src/golden_csv.py
    accept-suggested` writes them directly, tagged `bulk_accepted: True` in
    both `golden_v1.jsonl` and `labeling_log.jsonl`. This means the
    resulting 0/150 override rate cannot be distinguished from "the
    pre-annotator was very accurate" vs. "the review was shallow" — that
    ambiguity is disclosed in the report rather than presented as a clean
    result. The 50 blind items (no suggestion exists to bulk-accept) still
    require `label_tui.py --blind-only` and are the only source of an
    actual measured override/anchoring signal in this golden set. Gave up:
    the ability to claim a fully independent, item-by-item human review of
    all 200 examples — the report says exactly which 150 weren't reviewed
    that way and why that matters for how much to trust the override rate.

17. **No agent framework, vector database, or orchestration layer.**
    Retrieval is a ~60-line BM25 index; the agent is one structured LLM call
    plus deterministic guardrails. The brief states candidates will be asked
    to explain and modify their own code live — every added dependency is
    something to defend in that conversation, and none of LangChain /
    LlamaIndex / a vector DB service solves a problem this dataset actually
    has at this scale. Gave up: looking more "sophisticated" on paper.
