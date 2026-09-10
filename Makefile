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

csv-export:
	uv run python -m src.golden_csv export

csv-import:
	uv run python -m src.golden_csv import

# Full pipeline against cached LLM outputs only. No network, no API keys.
reproduce:
	uv run python -m src.eval.run_eval

# Full pipeline against live APIs, populating the cache.
live:
	uv run python -m src.eval.run_eval --live

demo:
	uv run python -m src.agent "$(MSG)"
