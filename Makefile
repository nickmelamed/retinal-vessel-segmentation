.DEFAULT_GOAL := help
.PHONY: help setup lock lock-check check-data lint test smoke ci agent-check

EXPORT := uv export --no-dev --no-hashes --no-emit-project --quiet

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-14s %s\n", $$1, $$2}'

# renv::restore() joins this target in phase 8, when the R report arrives.
setup:  ## Install the locked environment and the git hooks
	uv sync --locked
	uv run pre-commit install

lock:  ## Update uv.lock and regenerate requirements.txt (the only way deps change)
	uv lock
	$(EXPORT) > requirements.txt

lock-check:  ## Fail if uv.lock or requirements.txt is out of date
	uv lock --check
	$(EXPORT) | diff -u requirements.txt - || (echo "requirements.txt is stale. Run make lock." && exit 1)

check-data:  ## Verify datasets against data/CHECKSUMS.sha256
	uv run python scripts/check_data.py

lint:  ## ruff, ruff format --check, mypy, sqlfluff
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run sqlfluff lint sql

test:  ## pytest with coverage
	uv run pytest

smoke:  ## End-to-end run on synthetic data
	uv run pytest -m smoke --no-cov -q

ci: lint test smoke lock-check  ## Exactly what CI runs

agent-check:  ## Fast checks the Claude Code Stop hook runs
	uv run ruff check . && uv run ruff format --check .
	uv run mypy src
	uv run pytest -x -q -m "not slow" --no-cov
	python3 scripts/agent/check_style.py .
	python3 scripts/agent/check_numbers.py README.md MODEL_CARD.md docs/one_pager.md --sources results/tables results/release
