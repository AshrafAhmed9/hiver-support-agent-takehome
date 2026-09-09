# Hiver support-agent take-home

This repository evaluates a small retrieval-grounded support drafting agent on one brand from the Customer Support on Twitter dataset. It classifies an inbound message, drafts a reply from earlier support examples, and either permits a public draft or escalates it with a reason.

The implementation is in progress. Results will be published only after the brand, taxonomy, thresholds, and final evaluation split are frozen. The project treats historical replies as examples of wording and general guidance, not proof that an issue was resolved or that the same action is currently allowed.

## Setup

Python 3.12 is required. The project uses `uv` to create an isolated environment:

```bash
uv sync --python 3.12 --all-groups
```

Run the current data-pipeline tests with:

```bash
uv run pytest
```

## Development sequence

1. Download and inspect the source data with `uv run python -m src.data fetch`.
2. Build the time-split, redacted candidate records with `uv run python -m src.data build`.
3. Select a brand from the recorded early-window evidence before opening final test examples.
4. Label training and development sets, freeze the policy, then label and evaluate the untouched golden set.

`make reproduce` will be added when frozen predictions, labels, and report figures exist. It will replay recorded outputs and recompute the published metrics without API keys.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the evaluation design and constraints.

