PYTHON ?= ./venv/bin/python
PIP := $(PYTHON) -m pip
PYTEST := PYTHONPATH=src $(PYTHON) -m pytest

.PHONY: bootstrap dep-check fmt fmt-check lint type test-fast test-full perf-smoke quality-guardrails check ci-fast ci-full

bootstrap:
	./tools/bootstrap_dev.sh

dep-check:
	$(PIP) check

fmt:
	$(PYTHON) -m black src tests

fmt-check:
	$(PYTHON) -m black --check src tests

lint:
	$(PYTHON) -m flake8 src

type:
	$(PYTHON) -m mypy src

test-fast:
	$(PYTEST) tests/unit -m "not slow and not network" -q

test-full:
	$(PYTEST) tests -m "not network" -v

perf-smoke:
	$(PYTHON) tools/perf_smoke.py

quality-guardrails:
	$(PYTHON) tools/quality_guardrails.py

check: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails

ci-fast: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails

ci-full: test-full
