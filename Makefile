PYTHON ?= ./venv/bin/python
PIP := $(PYTHON) -m pip
PYTEST := PYTHONPATH=src $(PYTHON) -m pytest
MIN_COVERAGE_GAIN ?= 0.005
MAX_BAD_RATIO_INCREASE ?= 0.002
TEST_FAST_PYTEST_FLAGS := -q

ifdef CI
TEST_FAST_PYTEST_FLAGS += --durations=20 --durations-min=0.1
endif

.PHONY: bootstrap dep-check fmt fmt-check lint type test-fast test-full perf-smoke quality-guardrails bootstrap-quality-guardrails visual-eval visual-eval-guardrails bootstrap-calibrate benchmark-validate benchmark-run benchmark-aggregate-only benchmark-matrix benchmark-recommend benchmark-compare-correction benchmark-classify-failures benchmark-profile-runtime benchmark-compare-runtime benchmark-recommend-human-guidance benchmark-analyze-agreement benchmark-sweep-agreement benchmark-run-bg benchmark-status benchmark-kill curated-open curated-canary-prewarm-sources curated-canary-guardrails curated-canary-compare curated-canary-eval curated-canary-experiment check ci-fast ci-full

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
	$(PYTEST) tests/unit -m "not slow and not network" $(TEST_FAST_PYTEST_FLAGS)

test-full:
	$(PYTEST) tests -m "not network" -v

perf-smoke:
	$(PYTHON) tools/perf_smoke.py

quality-guardrails:
	$(PYTHON) tools/quality_guardrails.py

bootstrap-quality-guardrails:
	$(PYTHON) tools/bootstrap_quality_guardrails.py

visual-eval:
	$(PYTHON) tools/run_visual_eval.py

visual-eval-guardrails:
	$(PYTHON) tools/visual_eval_guardrails.py

bootstrap-calibrate:
	$(PYTHON) tools/bootstrap_calibrate_thresholds.py

benchmark-validate:
	$(PYTHON) tools/validate_benchmark_manifest.py

benchmark-run:
	$(PYTHON) tools/run_benchmark_suite.py

benchmark-aggregate-only:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required (existing benchmark run directory)"; exit 2)
	$(PYTHON) tools/run_benchmark_suite.py --resume-run-dir "$(RUN_DIR)" --aggregate-only

benchmark-matrix:
	$(PYTHON) tools/run_benchmark_strategy_matrix.py

benchmark-recommend:
	$(PYTHON) tools/recommend_benchmark_defaults.py

benchmark-compare-correction:
	@test -n "$(BASELINE)" || (echo "BASELINE is required (run dir or benchmark_report.json path)"; exit 2)
	@test -n "$(CORRECTED)" || (echo "CORRECTED is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/compare_benchmark_correction.py --baseline "$(BASELINE)" --corrected "$(CORRECTED)" \
		$(if $(ASSERT_TRADEOFF),--assert-agreement-tradeoff --min-coverage-gain "$(MIN_COVERAGE_GAIN)" --max-bad-ratio-increase "$(MAX_BAD_RATIO_INCREASE)",)

benchmark-classify-failures:
	@test -n "$(REPORT)" || (echo "REPORT is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/classify_alignment_failures.py --report "$(REPORT)" $(if $(MATCH),--match "$(MATCH)",)

benchmark-profile-runtime:
	@test -n "$(REPORT)" || (echo "REPORT is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/profile_benchmark_runtime.py --report "$(REPORT)" $(if $(TOP),--top "$(TOP)",)

benchmark-compare-runtime:
	@test -n "$(BASELINE)" || (echo "BASELINE is required (run dir or benchmark_report.json path)"; exit 2)
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/compare_benchmark_runtime.py --baseline "$(BASELINE)" --candidate "$(CANDIDATE)" \
		$(if $(TOP),--top "$(TOP)",) \
		$(if $(ONLY_POSITIVE),--only-positive-delta,) \
		$(if $(OUT_JSON),--output-json "$(OUT_JSON)",) \
		$(if $(OUT_MD),--output-md "$(OUT_MD)",)

benchmark-recommend-human-guidance:
	@test -n "$(REPORT)" || (echo "REPORT is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/recommend_human_guidance_tasks.py --report "$(REPORT)" $(if $(TOP),--top "$(TOP)",) \
		$(if $(MIN_PRIORITY),--min-priority "$(MIN_PRIORITY)",)

benchmark-analyze-agreement:
	@test -n "$(BASELINE_LABEL)" || (echo "BASELINE_LABEL is required (e.g. base)"; exit 2)
	@test -n "$(BASELINE)" || (echo "BASELINE is required (run dir or benchmark_report.json path)"; exit 2)
	@test -n "$(CANDIDATE_A_LABEL)" || (echo "CANDIDATE_A_LABEL is required (e.g. relax)"; exit 2)
	@test -n "$(CANDIDATE_A)" || (echo "CANDIDATE_A is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/analyze_agreement_tradeoffs.py --baseline "$(BASELINE_LABEL)=$(BASELINE)" \
		--candidate "$(CANDIDATE_A_LABEL)=$(CANDIDATE_A)" \
		$(if $(CANDIDATE_B_LABEL),--candidate "$(CANDIDATE_B_LABEL)=$(CANDIDATE_B)",) \
		$(if $(MIN_COVERAGE_GAIN),--min-coverage-gain "$(MIN_COVERAGE_GAIN)",) \
		$(if $(MAX_BAD_RATIO_INCREASE),--max-bad-ratio-increase "$(MAX_BAD_RATIO_INCREASE)",)

benchmark-sweep-agreement:
	@test -n "$(BASELINE)" || (echo "BASELINE is required (run dir or benchmark_report.json path)"; exit 2)
	@test -n "$(TEXT_SIM_VALUES)" || (echo "TEXT_SIM_VALUES is required (e.g. 0.60,0.58)"; exit 2)
	@test -n "$(TOKEN_OVERLAP_VALUES)" || (echo "TOKEN_OVERLAP_VALUES is required (e.g. 0.50,0.48)"; exit 2)
	$(PYTHON) tools/sweep_agreement_thresholds.py --baseline "$(BASELINE)" \
		--text-sim-values "$(TEXT_SIM_VALUES)" \
		--token-overlap-values "$(TOKEN_OVERLAP_VALUES)" \
		$(if $(RUN_ID_PREFIX),--run-id-prefix "$(RUN_ID_PREFIX)",) \
		$(if $(MIN_COVERAGE_GAIN),--min-coverage-gain "$(MIN_COVERAGE_GAIN)",) \
		$(if $(MAX_BAD_RATIO_INCREASE),--max-bad-ratio-increase "$(MAX_BAD_RATIO_INCREASE)",) \
		$(if $(OFFLINE),--offline,)

benchmark-run-bg:
	./tools/run_benchmark_suite_bg.sh

benchmark-status:
	$(PYTHON) tools/benchmark_status.py

benchmark-kill:
	./tools/kill_benchmark_suites.sh

curated-open:
	@test -n "$(MATCH)" || (echo "MATCH is required (e.g. MATCH=\"Con Calma\")"; exit 2)
	PYTHONPATH=src $(PYTHON) tools/curated_clip_helper.py --match "$(MATCH)" --open-editor

curated-canary-prewarm-sources:
	$(PYTHON) tools/prewarm_lyrics_source_cache.py --manifest benchmarks/benchmark_songs.yaml --match "Blinding Lights|Derniere danse|Mi Gente|DESPECHA" --max-songs 4

curated-canary-guardrails:
	@test -f benchmarks/results/latest.json || (echo "latest benchmark pointer missing for curated canary guardrails"; exit 2)
	$(PYTHON) tools/main_benchmark_guardrails.py --skip-benchmark --guardrails-json benchmarks/curated_canary_guardrails.json --report-json "$$(cat benchmarks/results/latest.json)"

curated-canary-compare:
	@test -n "$(BASELINE)" || (echo "BASELINE is required (run dir or benchmark_report.json path)"; exit 2)
	@test -n "$(CORRECTED)" || (echo "CORRECTED is required (run dir or benchmark_report.json path)"; exit 2)
	$(PYTHON) tools/compare_benchmark_correction.py --baseline "$(BASELINE)" --corrected "$(CORRECTED)" \
		$(if $(ASSERT_TRADEOFF),--assert-agreement-tradeoff --min-coverage-gain "$(MIN_COVERAGE_GAIN)" --max-bad-ratio-increase "$(MAX_BAD_RATIO_INCREASE)",)

curated-canary-eval:
	$(PYTHON) tools/run_benchmark_suite.py --offline --gold-root benchmarks/gold_set_candidate/20260305T231015Z --match "Blinding Lights|Derniere danse|Mi Gente|DESPECHA" --max-songs 4 $(if $(RUN_ID),--run-id "$(RUN_ID)",) $(EXTRA_ARGS)
	@test -f benchmarks/results/latest.json || (echo "latest benchmark pointer missing after curated canary eval"; exit 2)
	$(PYTHON) tools/main_benchmark_guardrails.py --skip-benchmark --guardrails-json benchmarks/curated_canary_guardrails.json --report-json "$$(cat benchmarks/results/latest.json)"
	$(PYTHON) tools/classify_alignment_failures.py --report "$$(cat benchmarks/results/latest.json)"
	$(if $(BASELINE),$(PYTHON) tools/compare_benchmark_correction.py --baseline "$(BASELINE)" --corrected "$$(cat benchmarks/results/latest.json)" $(if $(ASSERT_TRADEOFF),--assert-agreement-tradeoff --min-coverage-gain "$(MIN_COVERAGE_GAIN)" --max-bad-ratio-increase "$(MAX_BAD_RATIO_INCREASE)",),)

curated-canary-experiment:
	@test -n "$(EXPERIMENT)" || (echo "EXPERIMENT is required (e.g. repeat_duration)"; exit 2)
	@test -n "$(RUN_PREFIX)" || (echo "RUN_PREFIX is required"; exit 2)
	$(PYTHON) tools/run_curated_experiment_harness.py --experiment "$(EXPERIMENT)" --run-prefix "$(RUN_PREFIX)" $(if $(LABEL),--label "$(LABEL)",) $(if $(MATCH),--match "$(MATCH)",) $(if $(MAX_SONGS),--max-songs "$(MAX_SONGS)",)

check: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails bootstrap-quality-guardrails benchmark-validate

ci-fast: dep-check fmt-check lint type test-fast perf-smoke quality-guardrails bootstrap-quality-guardrails benchmark-validate

ci-full: test-full
