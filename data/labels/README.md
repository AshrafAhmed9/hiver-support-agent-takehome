# AI-assigned reference labels

These labels are assigned by an AI model, not a human. Each record contains
`label_source: "ai"`, `human_reviewed: false`, the provider/model ID, assignment
timestamp, input/protocol/prompt hashes, intent, routing rationale and uncertainty.
The manifest records output hashes, counts and label distributions. The hosted
model ID is recorded; an immutable provider weight revision is not available.

## Inputs and sampling

- `train_ai_v1.jsonl`: 150 seeded training examples for classifier fitting.
- `dev_ai_v1.jsonl`: 60 seeded development examples for development diagnostics.
- `golden_ai_v1.jsonl`: the existing 200 evaluation candidates, preserved by ID.

The training/development sampler excludes selected IDs and exact duplicate customer
texts across those samples and the evaluation candidates. It uses one record per
existing component identifier within each of train and dev. The existing data
extractor does not establish full connected conversation components: no claim of
complete conversation isolation or corpus-wide near-duplicate exclusion is made.

The existing evaluation candidates were selected using nearest-centroid pseudo-
intent strata and confidence extremes, not uniform random sampling. They are a
challenge set; they cannot support a claim about ordinary support volume. Original
suggestions and pseudo-labels remain in the legacy candidates file for provenance
but are excluded from the AI annotator's inputs, along with future brand replies.
Only customer text and supplied prior context are passed to the labeling model.
Some prior turns have unknown timestamps or missing surrounding context; the
annotator is instructed to mark uncertainty rather than infer missing details.

## Reproduce or inspect

```bash
make label-ai
make check-labels   # validates complete labels offline, no API credentials
```

This resumes the main run with Qwen via Groq (`GROQ_API_KEY` for live calls).
The CLI also accepts `--provider gemini --model MODEL_ID` (`GEMINI_API_KEY`).
An initial Qwen run hit Groq's 8,000-token/minute limit; its 25 partial training
labels are preserved in `archive/train_qwen_partial.jsonl` and are excluded from
the main labeled sets. Gemini 2.5 Flash then labeled 150 training and 30 development
items before reaching its 20-request daily quota. The run continues with Gemini
3 Flash Preview, retaining the model ID on each item and model counts per set in
the manifest. After that model also reached its daily quota, Qwen completed the
remaining challenge labels. The mixed-model sets introduce annotator variation;
do not present these references as a uniform or independently calibrated standard.
The Gemini-assigned portion shares a model family with the configured Gemini reply judge:
future agreement between those models would not be independent validation.
Subsequent runs resume validated saved labels and reject changed input or protocol
hashes. Model changes are retained explicitly per item. Raw request/response text
is recorded in `artifacts/llm_cache.jsonl`.
Inspect `uncertain: true` and the supplied reason when considering later review.
The codebook in `data/golden/codebook.md` defines the taxonomy. The prompt adds
explicit routing rules and boundary clarifications; its hash identifies that
combined protocol. No suggestions are silently promoted to human labels.

## Interpretation

`should_escalate: false` means an adequate public reply may be possible within the
stated capability policy. It does not certify a generated reply, demonstrate
resolution, or verify current Spotify policy. Self-reported uncertainty is an
annotation aid, not a calibrated error probability.

Some model outputs violate the fixed rule that `other` and
`not_a_support_request` route to review. Finalization applies that rule with
`routing_label_source: "deterministic_policy"` and a `policy_adjustments` entry.
Every item retains `raw_model_label`, including the original routing rationale.
Other labels are unchanged. The manifest counts these adjustments; they are not
human corrections or evidence of model accuracy.

These are model-generated reference labels, not independently established ground
truth. Comparisons against them measure model agreement and can share model bias.
There is no human labeling or judge–human agreement study in these artifacts.
Reply-quality ratings require actual generated drafts; no such rating pool existed
when this labeling run started. Do not manufacture reply ratings or describe a
future AI rater comparison as human agreement.

The user authorized AI labeling with this disclosure. The original pasted brief
requested human labels; this repository does not independently verify a change to
that requirement. These files accurately describe what was done.
