# Next Session TODO (Quality / Metrics / Efficiency)

## 0. Human-guided correction loop (new priority track)
- [x] Add fast micro-adjustment hotkeys in Gold Timing Editor (fine/coarse nudge).
- [x] Add optional snap-to-audio-onset anchors in Gold Timing Editor for word/line correction.
- [x] Add line/word "jump-to-next-anchor" actions to reduce repetitive drag edits.
- [x] Add correction-session telemetry (edits per minute, undo rate, snap usage) to quantify UX speedups.
- [x] Add benchmark path that compares raw auto-alignment vs post-human-correction delta on agreement/timing metrics.
- [x] Add benchmark-driven recommendation tool to prioritize manual correction queue per run.

## 1. Highest-impact quality work
- [ ] Improve agreement coverage from ~0.25 toward >=0.35 without degrading start-error metrics.
- [x] Investigate low-performing songs under current guardrails strategy (`hybrid_whisper`):
  - [x] `Dua Lipa - Levitating` (provisional classification: sparse lexical comparability)
  - [x] `Queen - Bohemian Rhapsody` (provisional classification: sparse lexical comparability)
  - [x] `Bruno Mars - Uptown Funk` (provisional classification: sparse lexical comparability)
  - [x] `Imagine Dragons - Believer` (provisional classification: sparse lexical comparability)
  - [x] `Indila - Derniere danse` (provisional classification: sparse lexical comparability)
- [x] For each, classify dominant failure mode: sparse lexical comparability vs timing drift vs repetition handling.
- Latest probe notes (main benchmark set, offline, `hybrid_whisper`):
  - baseline refresh `20260305T212738Z` (`text_sim=0.64`, `token_overlap=0.55`) -> agreement_cov `0.307`, p95 `1.089`, bad_ratio `0.045`
  - `text_sim=0.60, token_overlap=0.50` -> agreement_cov `0.331`, p95 `1.117`, bad_ratio `0.049`
  - `text_sim=0.58, token_overlap=0.48` -> agreement_cov `0.332`, p95 `1.117`, bad_ratio `0.049`
  - `text_sim=0.55, token_overlap=0.45` -> agreement_cov `0.337`, p95 `1.178`, bad_ratio `0.051`
  - deeper sweep (`20260305T_sweep_deeper_*`):
    - `text_sim=0.56` (token overlap `0.50/0.46/0.42`) -> agreement_cov `0.336`, p95 `1.180`, bad_ratio `0.051`
    - `text_sim=0.52` (token overlap `0.50/0.46/0.42`) -> agreement_cov `0.341`, p95 `1.178`, bad_ratio `0.051`
  - focused probe `text_sim=0.50, token_overlap=0.40` (`20260305T_probe_ts50_to40`) -> agreement_cov `0.346`, p95 `1.194`, bad_ratio `0.054`
  - focused probe `text_sim=0.48, token_overlap=0.38` (`20260305T_probe_ts48_to38`) -> agreement_cov `0.352`, p95 `1.268`, bad_ratio `0.057`
    - reaches >=0.35 coverage target but with unacceptable timing/bad-ratio regression under current guard.
  - near-baseline sweep (`20260305T_sweep_near_base_*`, token overlap fixed `0.55`):
    - `text_sim=0.63` -> agreement_cov `0.314`, p95 `1.129`, bad_ratio `0.047` (guard pass)
    - `text_sim=0.62` -> agreement_cov `0.317`, p95 `1.125`, bad_ratio `0.047` (guard pass)
    - `text_sim=0.61` -> agreement_cov `0.320`, p95 `1.122`, bad_ratio `0.047` (guard pass, best score in sweep)
  - Recalibration decisions:
    - set default `Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM` to `0.61` (`token_overlap` remains `0.55`) based on full 10-song guard-pass near-baseline sweep.
    - post-adaptive-rescue probe `text_sim=0.58, token_overlap=0.48` (`20260305T_probe_post_rescue_ts58_to48`) also guard-passes:
      - agreement_cov `0.334`, p95 `1.117`, bad_ratio `0.049` vs baseline `20260305T220734Z` delta: coverage `+0.0101`, bad_ratio `+0.0017`, p95 `-0.0048`
    - promoted default `Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM` to `0.58`.
    - default-threshold validation run `20260305T221248Z` (`text_sim=0.58`, overlap default `0.55`):
      - agreement_cov `0.327`, p95 `1.119`, bad_ratio `0.047` (guardrails run `OK`).
    - token-overlap probe (`text_sim=0.58`) vs baseline `20260305T221623Z`:
      - overlap `0.52` (`20260305T_probe_ts58_to52`) -> no material change
      - overlap `0.50` (`20260305T_probe_ts58_to50`) -> guard-pass gain: coverage `+0.0067`, bad_ratio `+0.0017`, p95 `-0.0025`
    - promoted default `Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP` to `0.50`.
    - default-threshold validation run `20260305T222326Z` (`text_sim=0.58`, overlap `0.50`):
      - agreement_cov `0.334`, p95 `1.117`, bad_ratio `0.049` (guardrails run `OK`).
  - adaptive-rescue refinements:
    - `line_word_count` gate `8 -> 6` run `20260305T220446Z` -> agreement_cov `0.322`, p95 `1.122`, bad_ratio `0.047`
    - `line_word_count` gate `6 -> 5` run `20260305T220734Z` -> agreement_cov `0.324`, p95 `1.122`, bad_ratio `0.047`
    - strict short-line rescue (`3-4` words, overlap>=`0.90`, delta<=`0.12s`, high local confidence) run `20260305T221623Z`:
      - agreement_cov `0.327`, p95 `1.119`, bad_ratio `0.047`
      - no additional aggregate gain beyond current `0.58` default baseline, but guardrail-safe.
    - net +0.004 absolute coverage lift vs initial `0.61/0.55` calibration run, with no bad-ratio regression.
  - Under current guard (`min_coverage_gain=0.005`, `max_bad_ratio_increase=0.002`), mild-to-moderate relaxations (`text_sim=0.58-0.63`, overlap `0.50-0.55`) can pass; aggressive relaxations do not.

## 2. Alignment pipeline improvements
- [x] Add stronger deterministic path-selection telemetry in lyrics pipeline:
  - [x] record when fallback mapping was attempted
  - [x] record why fallback was accepted/rejected
  - [x] surface per-song selection in benchmark summaries
- [x] Replace heuristic-only selection with score-based chooser using stable internal metrics.
- [x] Evaluate if any legacy heuristics should be removed/optionalized when they harm measured quality.

## 3. Agreement diagnostics reliability
- [x] Expand text normalization for agreement matching (contractions/apostrophes/repeated fillers) with tests.
- [x] Add a benchmark assertion that agreement-coverage gains do not materially raise agreement-bad ratio.
- [x] Add per-song agreement comparability report (matched/eligible lines + skip reasons).
- [x] Add cross-run agreement tradeoff analysis tool for baseline/candidate report ranking and guard checks.
- [x] Add agreement-threshold sweep runner that executes candidate benchmark runs and auto-generates tradeoff analysis.

## 4. Benchmark/guardrail hardening
- [x] Re-run full strategy matrix after pipeline updates and compare against:
  - [x] `20260305Tmatrix_full-*`
  - [x] `20260305T064432Z`
  - [x] `20260305T_agreement_tune_hybrid_whisper`
- [x] Recalibrate guardrail thresholds only if new quality gains hold for full 10-song set.
- [x] Ensure `main_benchmark_guardrails.py` output remains warning-clean except known reference-divergence cases.

## 5. Efficiency work
- [x] Profile slowest songs in `hybrid_whisper` to reduce alignment wall time.
- [x] Cache/reuse expensive intermediate computations across candidate alignment passes.
- [x] Reduce duplicated work in benchmark runs when only diagnostics logic changes.
- [x] Add baseline/candidate runtime-delta comparison tooling with schema-aware phase comparability.
- [x] Add runtime-delta triage filters (`--top`, `--only-positive-delta`) for quicker regression isolation.
- [x] Make runtime-delta suite elapsed summary aggregate-only-safe (`total` vs `executed` song elapsed deltas).
- [x] Add runtime-delta comparability warnings (phase non-comparability + aggregate-only elapsed divergence).

## 6. Code quality / tech debt
- [x] Continue reducing complexity hotspots in `tools/run_benchmark_suite.py` (incremental extraction).
- [x] Extract CLI threshold validation + manifest filtering/aggregate-only scoping into pure helpers with focused tests.
- [x] Extract suite-elapsed/report assembly helpers from `run_benchmark_suite.main()` with focused helper tests.
- [x] Extract song execution/resume loop into `_collect_song_results()` and runner env setup into `_build_runner_env()`.
- [x] Extract final benchmark console summary printing into `_print_run_summary()`.
- [x] Extract warning/exit-policy orchestration into `_compute_run_warnings()` + `_determine_exit_code()`.
- [x] Extract final report persistence into `_persist_final_report_outputs()` with focused file-output test.
- [x] Extract pre-run context setup into `_prepare_run_context()` and route `main()` through it.
- [x] Extract cached-reuse and single-song execution helpers from `_collect_song_results()`.
- [x] Extract song-result append/checkpoint path into `_append_result_and_checkpoint()` helper with unit test.
- [x] Extract per-song collection flow into `_collect_single_song_result()` with fail-fast/control-flow tests.
- [x] Reduce agreement text normalization complexity via `_expand_agreement_token()` extraction.
- [x] Reduce complexity hotspots count in `run_benchmark_suite.py` from 7 -> 2 via warning/reference-divergence/lexical/triage helper extraction.
- [x] Keep files under quality-guardrail size limits while refactoring.
- [x] Add focused tests for new extraction units rather than broad integration expansions.

## 7. CI/CD checklist before every push
- [x] `black --check src tests`
- [x] `flake8 src/y2karaoke --count --max-complexity=15 --max-line-length=127 --statistics`
- [x] `mypy src`
- [x] targeted unit tests for touched modules
- [x] `python tools/quality_guardrails.py`
- [x] if benchmark logic changed: run `tools/main_benchmark_guardrails.py`

## 8. Current baseline snapshot (for quick context)
- Guardrail strategy: `hybrid_whisper`
- Guardrail timing-quality floor: `0.70`
- Latest strong run: `benchmarks/results/20260305T_agreement_tune_hybrid_whisper`
  - timing_quality_score_line_weighted_mean: `0.7818`
  - dtw_line_coverage_line_weighted_mean: `0.9257`
  - dtw_word_coverage_line_weighted_mean: `0.8511`
  - agreement_coverage_ratio_mean: `0.2523`
  - agreement_start_p95_abs_sec_mean: `1.051`
