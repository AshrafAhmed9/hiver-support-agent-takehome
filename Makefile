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

# Explicitly AI-assigned references; never human labels.
label-ai:
	uv run python -m src.ai_label --provider groq --model qwen/qwen3.8-27b --batch-size 5 --max-output-tokens 1000

check-labels:
	uv run python -m src.ai_label --validate

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
