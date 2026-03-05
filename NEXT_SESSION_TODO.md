# Next Session TODO (Post-0.35 Baseline Refresh)

## 0. Current Baseline (frozen reference)
- Latest strong run: `benchmarks/results/20260305T231015Z`
- Strategy: `hybrid_whisper`
- Current default agreement thresholds:
  - `Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM=0.58`
  - `Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP=0.50`
- Key aggregate metrics:
  - `agreement_coverage_ratio_total=0.3524`
  - `agreement_start_p95_abs_sec_mean=1.0675`
  - `agreement_bad_ratio_mean=0.049`
  - `timing_quality_score_line_weighted_mean=0.7907`
  - `dtw_line_coverage_line_weighted_mean=0.9342`
  - `dtw_word_coverage_line_weighted_mean=0.8522`

## 1. Primary Quality Goals
- [ ] Reduce benchmark warning pressure while preserving >=`0.35` agreement coverage total.
- [ ] Improve `agreement_start_p95_abs_sec_mean` from ~`1.07` toward <=`0.95` without reducing coverage.
- [ ] Keep `agreement_bad_ratio_mean` <=`0.050` while improving timing quality.
- [ ] Raise `agreement_coverage_ratio_mean` from current ~`0.329` toward >=`0.350`.

## 2. Human-Guided Correction Execution
- [x] Generate current correction handoff artifacts:
  - `benchmarks/results/human_guidance_tasks_20260305T231015Z.md`
  - `benchmarks/results/human_guidance_ready_20260305T231015Z.tsv`
- [ ] Run a focused manual-correction pass for current top priority songs:
  - `J Balvin - Mi Gente`
  - `Indila - Derniere danse`
  - `Bruno Mars - Uptown Funk`
  - `ROSALIA - DESPECHA`
  - `The Weeknd - Blinding Lights`
- [ ] Produce an auto-vs-corrected benchmark comparison using `tools/compare_benchmark_correction.py`.
- [ ] Quantify human edit ROI:
  - coverage delta
  - p95 delta
  - bad-ratio delta
  - edits-per-minute / snap-usage from editor telemetry

## 3. Pipeline Improvement Goals
- [ ] Investigate large-anchor-span mismatch behavior (especially songs with long noisy anchor segments).
- [ ] Add/validate another conservative agreement rescue that improves comparability only under tight timing constraints.
- [ ] Add a regression test fixture for long noisy `nearest_segment_start_text` anchors (Blinding Lights-like behavior).

## 4. Gold Set and Benchmark Data Quality
- [ ] Increase gold comparable-word coverage from ~`0.747` toward >=`0.800`.
- [ ] Prioritize gold-set expansion for songs with highest human-guidance priority and poor comparability.
- [ ] Keep gold updates traceable (song-level changelog note per rebaseline).

## 5. Runtime and Efficiency
- [ ] Keep full 10-song offline benchmark wall-time <=`110s` while applying quality changes.
- [ ] Track runtime regressions when agreement diagnostics change (candidate vs baseline runtime diff report).
- [ ] Continue avoiding recomputation for diagnostics-only iterations.

## 6. Tooling and UX Improvements
- [x] Extend human-guidance report with direct “first 2 suggested correction targets” per song (line index + reason).
- [x] Add optional JSON schema validation for `human_guidance_tasks.json` artifacts.
- [x] Add a compact “ready-to-edit” export list for editor workflow handoff.

## 7. CI/CD Checklist (reset for next push cycle)
- [ ] `black --check src tests`
- [ ] `flake8 src/y2karaoke --count --max-complexity=15 --max-line-length=127 --statistics`
- [ ] `mypy src`
- [ ] targeted unit tests for touched modules
- [ ] `python tools/quality_guardrails.py`
- [ ] if benchmark logic changed: run `tools/main_benchmark_guardrails.py`

## 8. Success Criteria for This New Queue
- [ ] At least one new run that keeps `agreement_coverage_ratio_total >= 0.35` and improves p95 meaningfully.
- [ ] At least one validated human-corrected run demonstrating measurable quality lift vs auto baseline.
- [ ] No net regression in bad-ratio or major runtime regressions.
