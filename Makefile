.DEFAULT_GOAL := help
.PHONY: help agent-check

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-14s %s\n", $$1, $$2}'

# Phase 0 extends this to ruff, mypy, pytest, and check_numbers.py (see PROGRESS.md).
agent-check:  ## Fast checks the Claude Code Stop hook runs
	python3 scripts/agent/check_style.py .
