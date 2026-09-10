.DEFAULT_GOAL := test

test:
	uv run pytest

lint:
	uv run ruff check src tests

fetch:
	uv run python -m src.data fetch

build:
	uv run python -m src.data build --brand SpotifyCares

shortlist:
	uv run python -m src.data shortlist

review-samples:
	uv run python -m src.data review-samples --brands AmazonHelp SpotifyCares Tesco Uber_Support Delta

taxonomy:
	uv run python -m src.taxonomy

candidates:
	uv run python -m src.sampling

preannotate:
	uv run python -m src.preannotate

# Requires a human. See IMPLEMENTATION_PLAN.md §7.
label:
	uv run python -m src.label_tui

# Only the 50 blind items (no suggestion) — the part that can't be bulk-accepted.
label-blind-only:
	uv run python -m src.label_tui --blind-only --resume

# Bulk-accepts the 150 non-blind suggestions verbatim, disclosed as
# bulk_accepted: True. Still requires label-blind-only for the other 50.
accept-suggested:
	uv run python -m src.golden_csv accept-suggested

csv-export:
	uv run python -m src.golden_csv export

csv-import:
	uv run python -m src.golden_csv import

rate-replies:
	uv run python -m src.label_tui --reply-rating

# Full pipeline against cached LLM outputs only. No network, no API keys.
# Not yet runnable end-to-end: blocked on the golden-set labelling session.
reproduce:
	uv run python -m src.eval.run_eval

# Full pipeline against live APIs, populating the cache.
live:
	uv run python -m src.eval.run_eval --live

demo:
	uv run python -m src.agent "$(MSG)"
