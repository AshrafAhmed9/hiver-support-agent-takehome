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

csv-export:
	uv run python -m src.golden_csv export

csv-import:
	uv run python -m src.golden_csv import

# Replays the actual live run (60 of the 250 golden items — Groq's 200k
# TPD quota doesn't stretch to 250 live generations in one sitting; see
# DECISIONS.md and REPORT.md's "misleading headline number" section)
# against cached LLM outputs only. No network, no API keys.
reproduce:
	uv run python -m src.eval.run_eval --sample-size 60

# Full pipeline against live APIs, populating the cache. Also defaults to
# 60 items for the same quota reason; pass --sample-size to change it.
live:
	uv run python -m src.eval.run_eval --live --sample-size 60

demo:
	uv run python -m src.agent "$(MSG)"
