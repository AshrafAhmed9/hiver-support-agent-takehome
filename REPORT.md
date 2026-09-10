# Evaluation report — work in progress

No end-to-end performance result has been measured yet. This document records
the labeling methodology and limitations; it is not the completed assignment report.

## Label provenance

Reference labels are AI-assigned using the model/provider recorded in
`data/labels/manifest.json` and each label row. The run uses Gemini 2.5 Flash and
Gemini 3 Flash Preview and Qwen 3.8 27B because of provider quotas; this adds
annotator variation. Exact counts by model are in the manifest.
Files contain 150 training, 60 development and 200
challenge-set examples when the run is complete. Every record carries
`label_source: "ai"`, `human_reviewed: false`, timestamps, input/protocol hashes,
intent, an escalation decision with rationale and uncertainty flags. See
[sampling and labeling notes](data/labels/README.md).

Fixed policy corrections for `other` and `not_a_support_request` are recorded
separately, preserving the original model decision in `raw_model_label`. They
are deterministic rule applications, not human adjudications.

Labels use only customer text and supplied prior context. Existing suggested
intents, sampling pseudo-labels, and future historical replies are excluded.
Routing labels describe whether general public assistance could be appropriate;
they do not score a generated draft or prove a resolution.

## What is misleading about my headline number?

There is no headline result yet. Any future evaluation against these labels
measures agreement with AI-assigned references, not human judgment. Annotation
errors and shared model biases may inflate or depress results. A different model
family does not eliminate that uncertainty. There are no human reply ratings,
so an AI-rater comparison must be called AI–AI agreement, not judge–human agreement.

The 200 existing candidates are selected by pseudo-intent strata and confidence
extremes. They form a challenge sample and cannot estimate ordinary inbound-volume
coverage. Complete conversation isolation and corpus-wide near-duplicate removal
have not yet been established in the current direct-exchange extraction. Some
context is incomplete; historical replies do not verify today's product policy
or private support outcomes.

## Remaining evidence

The full report still needs a working evaluation run against both baselines,
reply-quality ratings on actual generated drafts, uncertainty intervals, five
observed failure modes, and next steps based on those results. No missing result
or human annotation is implied by the existence of the AI label files.
