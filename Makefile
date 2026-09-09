.DEFAULT_GOAL := test

test:
	uv run pytest

lint:
	uv run ruff check src tests

fetch:
	uv run python -m src.data fetch

build:
	uv run python -m src.data build

shortlist:
	uv run python -m src.data shortlist

review-samples:
	uv run python -m src.data review-samples --brands AmazonHelp SpotifyCares Tesco Uber_Support Delta
