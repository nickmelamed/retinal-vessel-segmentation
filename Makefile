.DEFAULT_GOAL := help
.PHONY: help setup lock lock-check check-data train evaluate verify-checkpoints lint test smoke mutate ci agent-check

EXPORT := uv export --no-dev --no-hashes --no-emit-project --quiet
VARIANT ?= baseline

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

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

train:  ## Resumable 5-fold CV for one config (make train VARIANT=baseline)
	uv run python -m retinal_vessels.train --config configs/$(VARIANT).yaml

evaluate:  ## Fill AUCs, Brier, and thin/thick sensitivity from saved predictions (RUN=<run_id>, default latest finished)
	uv run python -m retinal_vessels.evaluate $(if $(RUN),--run-id $(RUN),)

verify-checkpoints:  ## Check downloaded checkpoints against the database (RUN=<run_id>, default latest; UNFINISHED=1 mid-run)
	uv run python scripts/verify_checkpoints.py $(if $(RUN),--run-id $(RUN),) $(if $(UNFINISHED),--allow-unfinished,)

lint:  ## ruff, ruff format --check, mypy, sqlfluff
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run sqlfluff lint sql

test:  ## pytest with coverage
	uv run pytest

smoke:  ## End-to-end run on synthetic data
	uv run pytest -m smoke --no-cov -q

mutate:  ## Mutation testing of the fold code (settings in pyproject.toml)
	rm -rf mutants
	uv run mutmut run
	uv run mutmut results

ci: lint test smoke lock-check  ## Exactly what CI runs

agent-check:  ## Fast checks the Claude Code Stop hook runs
	uv run ruff check . && uv run ruff format --check .
	uv run mypy src
	uv run pytest -x -q -m "not slow" --no-cov
	python3 scripts/agent/check_style.py .
	python3 scripts/agent/check_numbers.py README.md MODEL_CARD.md docs/one_pager.md --sources results/tables results/release
