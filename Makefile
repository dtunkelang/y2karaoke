PYTHON ?= ./venv/bin/python
PIP := $(PYTHON) -m pip
PYTEST := PYTHONPATH=src $(PYTHON) -m pytest

.PHONY: bootstrap dep-check fmt fmt-check lint type test-fast test-full perf-smoke quality-guardrails bootstrap-quality-guardrails visual-eval visual-eval-guardrails bootstrap-calibrate benchmark-validate benchmark-run benchmark-matrix benchmark-recommend benchmark-run-bg benchmark-status benchmark-kill check ci-fast ci-full

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

bootstrap-quality-guardrails:
	$(PYTHON) tools/bootstrap_quality_guardrails.py

visual-eval:
	$(PYTHON) run_visual_eval.py

visual-eval-guardrails:
	$(PYTHON) tools/visual_eval_guardrails.py

bootstrap-calibrate:
	$(PYTHON) tools/bootstrap_calibrate_thresholds.py

benchmark-validate:
	$(PYTHON) tools/validate_benchmark_manifest.py

benchmark-run:
	$(PYTHON) tools/run_benchmark_suite.py

benchmark-matrix:
	$(PYTHON) tools/run_benchmark_strategy_matrix.py

benchmark-recommend:
	$(PYTHON) tools/recommend_benchmark_defaults.py

benchmark-run-bg:
	./tools/run_benchmark_suite_bg.sh

benchmark-status:
	$(PYTHON) tools/benchmark_status.py

benchmark-kill:
	./tools/kill_benchmark_suites.sh

check: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails bootstrap-quality-guardrails benchmark-validate

ci-fast: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails bootstrap-quality-guardrails benchmark-validate

ci-full: test-full
